package com.example.deepshield.data.repoIMPL

import android.content.Context
import android.graphics.Bitmap
import android.util.Log
import androidx.core.net.toUri
import io.ktor.client.request.*
import com.example.deepshield.Constants.Constants
import com.example.deepshield.data.KtorClient.KtorClient
import com.example.deepshield.data.Response.AudioResponse
import com.example.deepshield.data.Response.DeepFakeVideoResponse
import com.example.deepshield.data.Response.GetFrameResponse
import com.example.deepshield.data.Response.GradCamResponse
import com.example.deepshield.data.Response.HeatMapResponse
import com.example.deepshield.data.Response.ImageResponse
import com.example.deepshield.data.Response.NewResponse
import com.example.deepshield.domain.Repository.Repository
import com.example.deepshield.domain.StateHandling.ApiResult
import io.ktor.client.HttpClient
import io.ktor.client.call.body
import io.ktor.client.request.forms.formData
import io.ktor.client.request.forms.submitFormWithBinaryData
import io.ktor.client.request.get
import io.ktor.client.statement.HttpResponse
import io.ktor.http.Headers
import io.ktor.http.HttpHeaders
import kotlinx.coroutines.flow.Flow
import kotlinx.coroutines.flow.flow
import java.io.File
import java.io.FileInputStream
import java.io.InputStream
import javax.inject.Inject

class RepositoryImpl @Inject constructor(private val httpClient: HttpClient):Repository {
        override suspend fun uploadVideoToDeepFakeServer(
        context: Context,
        videoUri: String
    ): Flow<ApiResult<DeepFakeVideoResponse>> = flow {
        emit(ApiResult.Loading)

        val inputStream: InputStream? = context.contentResolver.openInputStream(videoUri.toUri())
        if (inputStream == null) {
            emit(ApiResult.Error("Failed to open video file"))
            return@flow
        }

        try {
            val response: HttpResponse = httpClient.submitFormWithBinaryData(
                url = "${Constants.BASE_URL}${Constants.VIDEO_ROUTE}",  // ✅ Corrected URL Concatenation
                formData = formData {
                    append("video",
                        inputStream.readBytes(),  // ✅ Correct way to read InputStream
                        Headers.build {
                            append(HttpHeaders.ContentType, "video/mp4")
                            append(HttpHeaders.ContentDisposition, "form-data; name=\"video\"; filename=\"video.mp4\"")
                        }
                    )
                }
            )

            val apiResponse:DeepFakeVideoResponse = response.body()
            Log.d("APIRESPONSE", "${apiResponse}")
            emit(ApiResult.Success(apiResponse))

        } catch (e: Exception) {
            emit(ApiResult.Error(e.message ?: "Unknown error"))
            Log.d("APIRESPONSE", "${e.message}")
        } finally {
            inputStream.close() // ✅ Close InputStream to avoid memory leaks
        }
    }

    override suspend fun getFrameFromServer(): Flow<ApiResult<Bitmap>> = flow {
        emit(ApiResult.Loading)
        try {
            // Perform Ktor GET request and parse response
            val response: GetFrameResponse = httpClient.get("").body()

            // Convert image_bytes to Bitmap
            response.toBitmap()?.let { bitmap ->
                emit(ApiResult.Success(bitmap))
            } ?: emit(ApiResult.Error("Failed to convert bytes to Bitmap"))

        } catch (e: Exception) {
            emit(ApiResult.Error(e.message ?: "Unknown error"))
        }
    }

    override suspend fun getHeatMapFromServer(): Flow<ApiResult<Bitmap>> = flow{
        emit(ApiResult.Loading)
        try {
            val response: HttpResponse =httpClient.get("${Constants.BASE_URL}${Constants.HEATMAP}")
            val apiResponse: HeatMapResponse = response.body()

            // Convert the response to a Bitmap
            val bitmap = apiResponse.toBitmap()
            emit(ApiResult.Success(bitmap!!))
        } catch (e: Exception) {
            emit(ApiResult.Error(e.message.toString()))
        }
    }

    override suspend fun getGradCamFromServer(): Flow<ApiResult<GradCamResponse>> =flow{
        emit(ApiResult.Loading)
        try {
            val response: HttpResponse = httpClient.get("${Constants.BASE_URL}${Constants.GRADCAM}")
            val apiResponse: GradCamResponse = response.body()
            emit(ApiResult.Success(apiResponse))
        }catch (e:Exception){
            emit(ApiResult.Error(e.message.toString()))
        }
    }

override suspend fun uploadAudioToDeepFakeServer(
    context: Context,
    audioUri: String // this is a *path*, not a content URI
): Flow<ApiResult<AudioResponse>> = flow {
    emit(ApiResult.Loading)

    val file = File(audioUri)
    val inputStream: InputStream? = if (file.exists()) FileInputStream(file) else null

    if (inputStream == null) {
        emit(ApiResult.Error("Failed to open audio file from path"))
        return@flow
    }

    try {
        val response: HttpResponse = httpClient.submitFormWithBinaryData(
            url = "${Constants.BASE_URL}${Constants.AUDIO_ROUTE}",
            formData = formData {
                append(
                    "file",
                    inputStream.readBytes(),
                    Headers.build {
                        append(HttpHeaders.ContentType, "audio/mpeg")
                        append(
                            HttpHeaders.ContentDisposition,
                            "form-data; name=\"file\"; filename=\"${file.name}\""
                        )
                    }
                )
            }
        )

        val apiResponse: AudioResponse = response.body()
        emit(ApiResult.Success(apiResponse))

    } catch (e: Exception) {
        Log.e("AUDIO_API_ERROR", e.message ?: "Unknown error", e)
        emit(ApiResult.Error(e.message ?: "Unknown error"))
    } finally {
        inputStream.close()
    }
}

    override suspend fun newsPrediction(claim: String): Flow<ApiResult<NewResponse>> =flow{
        emit(ApiResult.Loading)
        try {
            val response: HttpResponse =httpClient.get("${Constants.BASE_URL}${Constants.NEWS_ROUTE}"){
                parameter("claim",claim)
            }

            val apiResponse: NewResponse = response.body()
            Log.d("PREDICTION", apiResponse.toString())
            emit(ApiResult.Success(apiResponse))
        }catch (e:Exception){
            emit(ApiResult.Error(e.message.toString()))
        }

    }



    override suspend fun imagePrediction(context: Context,imageUri: String): Flow<ApiResult<ImageResponse>> =flow{
        emit(ApiResult.Loading)

        val inputStream: InputStream? = context.contentResolver.openInputStream(imageUri.toUri())
        if (inputStream == null) {
            emit(ApiResult.Error("Failed to open image file"))
            return@flow
        }

        try {
            val response: HttpResponse =httpClient.submitFormWithBinaryData(
                url = "${Constants.BASE_URL}${Constants.IMAGE_ROUTE}",
                formData = formData {
                    append(
                        "image",
                        inputStream.readBytes(),
                        Headers.build {
                            append(HttpHeaders.ContentType, "image/jpeg") // or image/png if needed
                            append(HttpHeaders.ContentDisposition, "form-data; name=\"image\"; filename=\"image.jpg\"")
                        }
                    )
                }
            )

            val apiResponse: ImageResponse = response.body()
            emit(ApiResult.Success(apiResponse))
        } catch (e: Exception) {
            emit(ApiResult.Error(e.message ?: "Unknown error"))
        } finally {
            inputStream.close()
        }
    }


}