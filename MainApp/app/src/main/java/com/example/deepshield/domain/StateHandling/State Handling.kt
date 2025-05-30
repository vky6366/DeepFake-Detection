package com.example.deepshield.domain.StateHandling

import android.graphics.Bitmap
import com.example.deepshield.data.Response.AudioResponse
import com.example.deepshield.data.Response.DeepFakeVideoResponse
import com.example.deepshield.data.Response.GradCamResponse
import com.example.deepshield.data.Song.Song

sealed class ApiResult<out T> {
    object Loading : ApiResult<Nothing>()
    data class Success<T>(val data: T) : ApiResult<T>()
    data class Error(val message: String) : ApiResult<Nothing>()
}
sealed class ResultState<out T>{
    object Loading: ResultState<Nothing>()
    data class Success<T>(val data: T): ResultState<T>()
    data class Error(val message: String): ResultState<Nothing>()
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
data class  HeatMapResponseState(
    val isLoading: Boolean = false,
    val data: Bitmap? = null,
    val error: String = ""
)

data class GradCamResponseState(
    val isLoading: Boolean = false,
    val data:GradCamResponse? = null,
    val error: String = ""
)

data class GetAllSongState(
    val isLoading: Boolean = false,
    val data: List<Song> = emptyList(),
    val error: String ? = null
)

data class AudioResponseState(
    val isLoading: Boolean = false,
    val data: AudioResponse? = null,
    val error: String = ""
)
