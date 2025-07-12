package org.example.project.domain.UseCase

import kotlinx.coroutines.flow.Flow
import org.example.project.data.ApiResponse.DeepFakeVideoResponse
import org.example.project.data.stateHandler.ApiResult
import org.example.project.domain.Repository

class UploadVideoToDeepFakeServerUseCase(private val repository: Repository) {
    operator suspend fun invoke(videoBytes: ByteArray): Flow<ApiResult<DeepFakeVideoResponse>> {
        return repository.uploadVideoToDeepFakeServer(videoBytes = videoBytes)
    }

}