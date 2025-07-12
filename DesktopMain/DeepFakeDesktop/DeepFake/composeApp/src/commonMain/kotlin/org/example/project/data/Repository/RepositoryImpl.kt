package org.example.project.data.Repository

import com.example.deepshield.Constants.Constants
import io.ktor.client.call.body
import io.ktor.client.request.forms.formData
import io.ktor.client.request.forms.submitFormWithBinaryData
import io.ktor.client.statement.HttpResponse
import io.ktor.http.Headers
import io.ktor.http.HttpHeaders
import kotlinx.coroutines.flow.Flow
import kotlinx.coroutines.flow.flow
import org.example.project.data.ApiResponse.AudioResponse
import org.example.project.data.ApiResponse.DeepFakeVideoResponse
import org.example.project.data.ApiResponse.ImageResponse
import org.example.project.data.stateHandler.ApiResult
import org.example.project.domain.Repository.Repository
import org.example.project.provideHttpClient

class RepositoryImpl() : Repository {
    override suspend fun uploadVideoToDeepFakeServer(
        videoBytes: ByteArray
    ): Flow<ApiResult<DeepFakeVideoResponse>> = flow {
        emit(ApiResult.Loading)
        try {
            val response: HttpResponse = provideHttpClient().submitFormWithBinaryData(
                url = "${Constants.BASE_URL}${Constants.VIDEO_ROUTE}",
                formData = formData {
                    append(
                        "video",
                        videoBytes,
                        Headers.build {
                            append(HttpHeaders.ContentType, "video/mp4")
                            append(HttpHeaders.ContentDisposition, "form-data; name=\"video\"; filename=\"video.mp4\"")
                        }
                    )
                }
            )

            val apiResponse: DeepFakeVideoResponse = response.body()
            emit(ApiResult.Success(apiResponse))

        } catch (e: Exception) {
            emit(ApiResult.Error(e.message ?: "Unknown error"))
        }
    }

    override suspend fun uploadAudioToDeepFakeServer(audioUrl: ByteArray): Flow<ApiResult<AudioResponse>> =flow{
        emit(ApiResult.Loading)
        try {
            val response: HttpResponse = provideHttpClient().submitFormWithBinaryData(
                url = "${Constants.BASE_URL}${Constants.VIDEO_ROUTE}",
                formData = formData {
                    append(
                        "audio",
                        audioUrl,
                        Headers.build {
                            append(HttpHeaders.ContentType, "audio/mpeg")
                            append(HttpHeaders.ContentDisposition, "form-data; name=\"file\"; filename=\"audio.mp3\"")
                        }
                    )
                }
            )

            val apiResponse: AudioResponse = response.body()
            emit(ApiResult.Success(apiResponse))

        } catch (e: Exception) {
            emit(ApiResult.Error(e.message ?: "Unknown error"))
        }

    }

    override suspend fun imagePrediction(imageUri: ByteArray): Flow<ApiResult<ImageResponse>> =flow{
        emit(ApiResult.Loading)

        try {
            val response: HttpResponse = provideHttpClient().submitFormWithBinaryData(
                url = "${Constants.BASE_URL}${Constants.IMAGE_ROUTE}",
                formData = formData {
                    append(
                        "image",
                        imageUri,
                        Headers.build {
                            append(HttpHeaders.ContentType, "image/jpeg") // or "image/png"
                            append(HttpHeaders.ContentDisposition, "form-data; name=\"image\"; filename=\"image.jpg\"")
                        }
                    )
                }
            )

            val apiResponse: ImageResponse = response.body()
            emit(ApiResult.Success(apiResponse))
        } catch (e: Exception) {
            emit(ApiResult.Error(e.message ?: "Unknown error"))
        }
    }
}
