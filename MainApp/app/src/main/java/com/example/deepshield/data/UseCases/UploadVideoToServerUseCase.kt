package com.example.deepshield.data.UseCases

import android.content.Context
import com.example.deepshield.data.Response.DeepFakeVideoResponse
import com.example.deepshield.domain.Repository.Repository
import com.example.deepshield.domain.StateHandling.ApiResult
import kotlinx.coroutines.flow.Flow
import javax.inject.Inject

class UploadVideoToServerUseCase @Inject constructor(private val repository: Repository) {
    suspend fun execute(context:Context ,videoUrl:String): Flow<ApiResult<DeepFakeVideoResponse>> {
        return  repository.uploadVideoToDeepFakeServer(context = context, videoUrl = videoUrl)
    }
}