package org.example.project.data.stateHandler

import org.example.project.data.ApiResponse.AudioResponse
import org.example.project.data.ApiResponse.DeepFakeVideoResponse
import org.example.project.data.ApiResponse.ImageResponse
import org.example.project.data.ApiResponse.NewResponse

sealed class ApiResult<out T> {
    object Loading : ApiResult<Nothing>()
    data class Success<T>(val data: T) : ApiResult<T>()
    data class Error(val message: String) : ApiResult<Nothing>()
}

data class DeepFakeVideoResponseState(
    val isLoading: Boolean = false,
    val data: DeepFakeVideoResponse? = null,
    val error: String = "")

data class AudioResponseState(
    val isLoading: Boolean = false,
    val data: AudioResponse? = null,
    val error: String = ""
)

data class ImageResponseState(
    val isLoading: Boolean = false,
    val data: ImageResponse? = null,
    val error: String = ""
)


data class NewsPredictionState(
    val isLoading: Boolean = false,
    val data: NewResponse? = null,
    val error: String = ""
)
