package org.example.project.data.stateHandler

import org.example.project.data.ApiResponse.DeepFakeVideoResponse

sealed class ApiResult<out T> {
    object Loading : ApiResult<Nothing>()
    data class Success<T>(val data: T) : ApiResult<T>()
    data class Error(val message: String) : ApiResult<Nothing>()
}

data class DeepFakeVideoResponseState(
    val isLoading: Boolean = false,
    val data: DeepFakeVideoResponse? = null,
    val error: String = "")