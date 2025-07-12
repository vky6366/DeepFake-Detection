package org.example.project.domain.UseCase

import kotlinx.coroutines.flow.Flow
import org.example.project.data.ApiResponse.AudioResponse
import org.example.project.data.stateHandler.ApiResult
import org.example.project.domain.Repository.Repository

class UploadAudioToDeepFakeServerUseCase(private val repository: Repository) {
    suspend operator fun invoke(audioUrl: ByteArray): Flow<ApiResult<AudioResponse>> {
        return repository.uploadAudioToDeepFakeServer(audioUrl = audioUrl)
    }
}