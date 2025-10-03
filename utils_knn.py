from torch.ao.quantization import quantize_dynamic
from torch import Tensor
from torch.nn import Linear, Module
from typing import Optional, Tuple
import torch


class LinearLayer(Module):
    """
    Wrapper around torch.nn.Linear for a bias-free linear layer.
    Used for quantizing matrix multiplication operations.
    """

    def __init__(self, in_dim: int, out_dim: int):
        super(LinearLayer, self).__init__()
        self.fc = Linear(in_dim, out_dim, bias=False)

    def forward(self, inp: Tensor) -> Tensor:
        return self.fc(inp)


def quantize_model(tensor: Tensor) -> Module:
    """
    Create and quantize a linear layer model from a tensor.

    Args:
        tensor: Input tensor to be used as weights
        transpose: Whether to transpose the tensor before using as weights

    Returns:
        Quantized model
    """
    layer = LinearLayer(tensor.shape[1], tensor.shape[0])
    layer.fc.weight.data = tensor
    layer.eval()

    return quantize_dynamic(
        model=layer, qconfig_spec={Linear}, dtype=torch.qint8, inplace=True
    )


@torch.jit.script
def compute_distance(
    data: Tensor, query: Tensor, metric: str, pre_transposed: bool = False
) -> Tensor:
    """
    Compute distance matrix between query and data points.

    Args:
        data: Data tensor of shape (N, D) or (D, N) if pre_transposed
        query: Query tensor of shape (M, D)
        metric: Distance metric ('angular', 'euclidean', or 'manhattan')
        pre_transposed: Whether data is already transposed

    Returns:
        Distance matrix of shape (M, N)
    """
    if metric == "euclidean":
        # Compute norms in FP32 for numerical stability
        if pre_transposed:
            # data is (D, N)
            data_norm = (data.float() ** 2).sum(dim=0)  # (N,)
            q_norm = (query.float() ** 2).sum(dim=1)  # (M,)
            dist = (
                q_norm.view(-1, 1)
                + data_norm.view(1, -1)
                - (2.0 * query @ data).float()
            )
        else:
            # data is (N, D)
            data_norm = (data.float() ** 2).sum(dim=1)  # (N,)
            q_norm = (query.float() ** 2).sum(dim=1)  # (M,)
            dist = (
                q_norm.view(-1, 1)
                + data_norm.view(1, -1)
                - (2.0 * query @ data.t()).float()
            )
    elif metric == "dotprod":
        if pre_transposed:
            dist = query @ data  # data already transposed
        else:
            dist = query @ data.t()
    elif metric == "cosine":
        if pre_transposed:
            # data is (D, N)
            dist = 1 - (query @ data) / (
                query.norm(dim=1, keepdim=True, p=2)
                * data.norm(dim=0, keepdim=True, p=2)
                + 1e-8
            )
        else:
            # data is (N, D)
            dist = 1 - (query @ data.t()) / (
                query.norm(dim=1, keepdim=True, p=2)
                * data.norm(dim=1, keepdim=True, p=2)
                + 1e-8
            )
    elif metric == "manhattan":
        if pre_transposed:
            # Transpose back for element-wise operations
            dist = (query[:, None, :] - data.t()[None, :, :]).abs().sum(dim=2)
        else:
            dist = (query[:, None, :] - data[None, :, :]).abs().sum(dim=2)
    else:
        raise NotImplementedError(f"Distance metric '{metric}' not supported.")

    return dist


class ChunkedKNN:
    """
    Efficient ChunkedKNN implementation that supports pre-transposed data.
    Optimized for the common use case where data is set once and queried multiple times.
    """

    def __init__(
        self,
        query_size: int,
        topk: int,
        distance_metric,
        match_device: torch.device,
        match_dtype: torch.dtype,
        quantized: bool = True,
        transpose_data: bool = True,
    ):
        """
        Initialize the ChunkedKNN indexer for batched k-nearest neighbors search.

        Args:
            query_size: Total number of query entries
            topk: Number of nearest neighbors to return
            distance_metric: Distance metric to use ('euclidean', 'manhattan', 'dotprod', or 'cosine')
            match_device: Device for computation
            match_dtype: Data type for computation (torch.float32, torch.float16)
            quantized: Whether to use quantized computation (default: True)
            transpose_data: Whether to transpose data for faster computation (default: True)

        """
        self.query_size = query_size
        self.topk = topk
        self.match_device = match_device
        self.distance_metric = distance_metric
        self.match_dtype = match_dtype
        self.quantized = quantized
        self.transpose_data = transpose_data
        self.big_better = self.distance_metric == "dotprod"

        # Validate and adjust settings
        if quantized and match_dtype != torch.float32 and match_device.type == "cpu":
            print(
                "CPU Quantization requires float32 data type. Forcing float32 match_dtype."
            )
            match_dtype = torch.float32

        if quantized and match_device.type == "cpu" and transpose_data:
            print(
                "Quantized computation on CPU does not support pre-transposed data. Disabling transpose_data."
            )
            self.transpose_data = False

        # Initialize results storage
        self.knn_indices = torch.empty(
            (self.query_size, self.topk),
            device=self.match_device,
            dtype=torch.int64,
        )
        self.knn_distances = torch.full(
            (self.query_size, self.topk),
            -torch.inf if self.big_better else torch.inf,
            device=self.match_device,
            dtype=self.match_dtype,
        )

        self.prepared_data = None
        self.curr_end = 0
        self.curr_start = 0

    def set_data_chunk(self, data_chunk: Tensor):
        """
        Set the current data chunk, optionally pre-transposing for efficiency.

        Args:
            data_chunk: Data chunk of shape (N, D)
        """
        # Convert to appropriate dtype and device
        data_chunk = data_chunk.to(self.match_dtype).to(self.match_device)

        # Update indices
        self.curr_end += data_chunk.shape[0]
        self.curr_start = self.curr_end - data_chunk.shape[0]

        # Quantize if needed
        if self.quantized and self.match_device.type == "cpu":
            self.prepared_data = quantize_model(data_chunk)
            self.prepared_data.eval()
        else:
            if self.transpose_data:
                self.prepared_data = data_chunk.mT
            else:
                self.prepared_data = data_chunk

    def query(
        self,
        query_batch: Tensor,
        start_idx: Optional[int] = None,
        end_idx: Optional[int] = None,
    ):
        """
        Perform k-nearest neighbors search on query batch.
        Can be used for both full dataset queries and chunk queries.

        Args:
            query_batch: Query tensor of shape (M, D)
            start_idx: Optional start index for updating specific portion of results
                       If None, assumes query_batch represents all queries
            end_idx: Optional end index (exclusive) for updating results
                     If None, computed as start_idx + query_batch.shape[0]
        """
        if self.prepared_data is None:
            raise ValueError("Data chunk not set. Call set_data_chunk first.")

        # Convert query to appropriate dtype and device
        query_batch = query_batch.to(self.match_dtype).to(self.match_device)

        # Determine indices for result update
        if start_idx is None:
            # Full dataset query
            start_idx = 0
            end_idx = self.query_size
        elif end_idx is None:
            # Chunk query with automatically computed end index
            end_idx = start_idx + query_batch.shape[0]

        if end_idx > self.query_size:
            raise ValueError(
                f"End index {end_idx} exceeds query size {self.query_size}"
            )

        # Compute KNN
        if self.quantized and self.match_device.type == "cpu":
            knn_dists_chunk, knn_inds_chunk = torch.topk(
                self.prepared_data(query_batch),
                self.topk,
                dim=1,
                largest=self.big_better,
                sorted=False,
            )
        else:
            # Use regular computation with pre-transposed data if configured
            dists = compute_distance(
                self.prepared_data,
                query_batch,
                self.distance_metric,
                self.transpose_data,
            )
            knn_dists_chunk, knn_inds_chunk = torch.topk(
                dists, self.topk, dim=1, largest=self.big_better, sorted=False
            )

        # Adjust indices to global space
        knn_inds_chunk += self.curr_start

        # Get current results for the chunk
        curr_indices = self.knn_indices[start_idx:end_idx]
        curr_distances = self.knn_distances[start_idx:end_idx]

        # Merge with existing results
        merged_distances = torch.cat((curr_distances, knn_dists_chunk), dim=1)
        merged_indices = torch.cat((curr_indices, knn_inds_chunk), dim=1)

        # Get overall topk
        topk_indices = torch.topk(
            merged_distances, self.topk, dim=1, largest=self.big_better, sorted=False
        )[1]

        # Update results
        self.knn_indices[start_idx:end_idx] = torch.gather(
            merged_indices, 1, topk_indices
        )
        self.knn_distances[start_idx:end_idx] = torch.gather(
            merged_distances, 1, topk_indices
        )

    def retrieve_topk(self) -> Tuple[Tensor, Tensor]:
        """
        Retrieve the top-k nearest neighbors indices and distances.

        Returns:
            Tuple of (indices, distances) sorted by distance
        """
        # Sort the topk indices and distances
        topk_indices = torch.topk(
            self.knn_distances, self.topk, dim=1, largest=self.big_better, sorted=True
        )[1]

        knn_indices = torch.gather(self.knn_indices, 1, topk_indices)
        knn_distances = torch.gather(self.knn_distances, 1, topk_indices)

        return knn_indices, knn_distances
