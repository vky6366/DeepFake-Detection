package com.example.deepshield.data.UseCases

import android.content.Context
import com.example.deepshield.data.Response.ImageResponse
import com.example.deepshield.domain.Repository.Repository
import com.example.deepshield.domain.StateHandling.ApiResult
import kotlinx.coroutines.flow.Flow
import javax.inject.Inject

class ImagePredictionUseCase @Inject constructor(private val repository: Repository){

    suspend operator fun invoke(context: Context,imageUri: String):Flow<ApiResult<ImageResponse>> {
        return repository.imagePrediction(context = context, imageUri = imageUri)
    }


}