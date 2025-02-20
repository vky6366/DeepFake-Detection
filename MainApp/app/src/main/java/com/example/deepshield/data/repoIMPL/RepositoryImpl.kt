package com.example.deepshield.data.repoIMPL

import android.content.Context
import androidx.core.net.toUri
import com.example.deepshield.Constants.Constants
import com.example.deepshield.data.KtorClient.KtorClient
import com.example.deepshield.data.Response.DeepFakeVideoResponse
import com.example.deepshield.domain.Repository.Repository
import com.example.deepshield.domain.StateHandling.ApiResult
import io.ktor.client.call.body
import io.ktor.client.request.forms.formData
import io.ktor.client.request.forms.submitFormWithBinaryData
import io.ktor.client.statement.HttpResponse
import io.ktor.http.Headers
import io.ktor.http.HttpHeaders
import kotlinx.coroutines.flow.Flow
import kotlinx.coroutines.flow.flow
import java.io.InputStream

class RepositoryImpl:Repository {
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
            val response: HttpResponse = KtorClient.client.submitFormWithBinaryData(
                url = "${Constants.BASE_URL}${Constants.VIDEO_ROUTE}",  // ✅ Corrected URL Concatenation
                formData = formData {
                    append("video",
                        inputStream.readBytes(),  // ✅ Correct way to read InputStream in Ktor
                        Headers.build {
                            append(HttpHeaders.ContentType, "video/mp4")
                            append(HttpHeaders.ContentDisposition, "form-data; name=\"video\"; filename=\"video.mp4\"")
                        }
                    )
                }
            )

            val apiResponse:DeepFakeVideoResponse = response.body()
            emit(ApiResult.Success(apiResponse))

        } catch (e: Exception) {
            emit(ApiResult.Error(e.message ?: "Unknown error"))
        } finally {
            inputStream.close() // ✅ Close InputStream to avoid memory leaks
        }
    }

}