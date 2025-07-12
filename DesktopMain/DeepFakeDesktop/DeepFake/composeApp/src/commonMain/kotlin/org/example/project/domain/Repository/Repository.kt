package org.example.project.domain.Repository

import kotlinx.coroutines.flow.Flow
import org.example.project.data.ApiResponse.AudioResponse
import org.example.project.data.ApiResponse.DeepFakeVideoResponse
import org.example.project.data.ApiResponse.ImageResponse
import org.example.project.data.stateHandler.ApiResult

interface Repository {
    suspend fun uploadVideoToDeepFakeServer( videoBytes: ByteArray): Flow<ApiResult<DeepFakeVideoResponse>>

    suspend fun uploadAudioToDeepFakeServer(audioUrl: ByteArray):Flow<ApiResult<AudioResponse>>

    suspend fun imagePrediction(imageUri: ByteArray):Flow<ApiResult<ImageResponse>>


}