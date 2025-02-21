package com.example.deepshield.domain.StateHandling

import android.graphics.Bitmap
import com.example.deepshield.data.Response.DeepFakeVideoResponse

sealed class ApiResult<out T> {
    object Loading : ApiResult<Nothing>()
    data class Success<T>(val data: T) : ApiResult<T>()
    data class Error(val message: String) : ApiResult<Nothing>()
}

data class DeepFakeVideoResponseState(
    val isLoading: Boolean = false,
    val data: DeepFakeVideoResponse? = null,
    val error: String = "")

data class FrameResponseState(
    val isLoading: Boolean = false,
    val bitmap: Bitmap? = null,
    val error: String = ""
)
