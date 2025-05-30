package com.example.deepshield.data.UseCases

import com.example.deepshield.data.Response.NewResponse
import com.example.deepshield.domain.Repository.Repository
import com.example.deepshield.domain.StateHandling.ApiResult
import kotlinx.coroutines.flow.Flow
import javax.inject.Inject

class NewsPredictionUseCase @Inject constructor(private val repository: Repository) {
    suspend operator fun invoke(claim: String): Flow<ApiResult<NewResponse>>{
        return repository.newsPrediction(claim = claim)
    }
}