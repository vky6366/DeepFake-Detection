package com.example.deepshield.data.UseCases

import android.graphics.Bitmap
import com.example.deepshield.domain.Repository.Repository
import com.example.deepshield.domain.StateHandling.ApiResult
import kotlinx.coroutines.flow.Flow
import javax.inject.Inject

class GetHeatMapUseCase  @Inject constructor(private val repository: Repository){

    suspend fun execute(): Flow<ApiResult<Bitmap>> {
        return repository.getHeatMapFromServer()

    }
}