package com.example.deepshield.data.UseCases

import com.example.deepshield.data.Response.GradCamResponse
import com.example.deepshield.domain.Repository.Repository
import com.example.deepshield.domain.StateHandling.ApiResult
import kotlinx.coroutines.flow.Flow
import javax.inject.Inject

class GetGradCamUseCase  @Inject constructor(private val repository: Repository){
    suspend fun execute(): Flow<ApiResult<GradCamResponse>> {
        return repository.getGradCamFromServer()
    }
}