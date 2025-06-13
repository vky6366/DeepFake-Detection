package com.example.deepshield.data.repoIMPL.TestRepo

import android.content.Context
import android.graphics.Bitmap
import com.example.deepshield.data.Response.AudioResponse
import com.example.deepshield.data.Response.DeepFakeVideoResponse
import com.example.deepshield.data.Response.GradCamResponse
import com.example.deepshield.data.Response.ImageResponse
import com.example.deepshield.data.Response.NewResponse
import com.example.deepshield.domain.Repository.Repository
import com.example.deepshield.domain.StateHandling.ApiResult
import kotlinx.coroutines.flow.Flow
import kotlinx.coroutines.flow.flow

class FakeRepository: Repository {

    //    val message: String,
    //    val prediction: String,
    //    val score: Double
    val deepFakeVideoResponse = DeepFakeVideoResponse(
        message = "FAKE",
        prediction = "Fake",
        score = 0.9
    )


    override suspend fun uploadVideoToDeepFakeServer(
        context: Context,
        videoUrl: String
    ): Flow<ApiResult<DeepFakeVideoResponse>> =flow{
        emit(ApiResult.Success(deepFakeVideoResponse))
    }

    override suspend fun getFrameFromServer(): Flow<ApiResult<Bitmap>> {
        TODO("Not yet implemented")
    }

    override suspend fun getHeatMapFromServer(): Flow<ApiResult<Bitmap>> {
        TODO("Not yet implemented")
    }

    override suspend fun getGradCamFromServer(): Flow<ApiResult<GradCamResponse>> {
        TODO("Not yet implemented")
    }

    override suspend fun uploadAudioToDeepFakeServer(
        context: Context,
        audioUrl: String
    ): Flow<ApiResult<AudioResponse>> {
        TODO("Not yet implemented")
    }

    override suspend fun newsPrediction(claim: String): Flow<ApiResult<NewResponse>> {
        TODO("Not yet implemented")
    }

    override suspend fun imagePrediction(
        context: Context,
        imageUri: String
    ): Flow<ApiResult<ImageResponse>> {
        TODO("Not yet implemented")
    }
}