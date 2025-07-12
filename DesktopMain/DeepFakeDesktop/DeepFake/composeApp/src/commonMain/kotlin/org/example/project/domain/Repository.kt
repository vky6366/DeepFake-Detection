package org.example.project.domain

import kotlinx.coroutines.flow.Flow
import org.example.project.data.ApiResponse.DeepFakeVideoResponse
import org.example.project.data.stateHandler.ApiResult

interface Repository {
    suspend fun uploadVideoToDeepFakeServer( videoBytes: ByteArray): Flow<ApiResult<DeepFakeVideoResponse>>
}