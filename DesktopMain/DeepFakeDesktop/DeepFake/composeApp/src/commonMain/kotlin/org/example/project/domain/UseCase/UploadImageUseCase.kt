package org.example.project.domain.UseCase

import kotlinx.coroutines.flow.Flow
import org.example.project.data.ApiResponse.ImageResponse
import org.example.project.data.stateHandler.ApiResult
import org.example.project.domain.Repository.Repository

class UploadImageUseCase(private val repository: Repository) {

    suspend operator fun invoke(imageUri: ByteArray): Flow<ApiResult<ImageResponse>> {
        return repository.imagePrediction(imageUri = imageUri)
    }
}