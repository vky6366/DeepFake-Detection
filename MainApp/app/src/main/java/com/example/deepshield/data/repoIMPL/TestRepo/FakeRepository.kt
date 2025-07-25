package com.example.deepshield.data.repoIMPL.TestRepo

import android.content.Context
import android.graphics.Bitmap
import com.example.deepshield.data.Response.AudioResponse
import com.example.deepshield.data.Response.DeepFakeVideoResponse
import com.example.deepshield.data.Response.GradCamResponse
import com.example.deepshield.data.Response.ImageResponse
import com.example.deepshield.data.Response.NewResponse
import com.example.deepshield.data.Response.Source
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
    val listOfString = listOf<String>("hello1","hello2","hello3")

    //    val focused_regions: List<String> = emptyList()
    val fakeGradCamResponse = GradCamResponse(
        focused_regions = listOfString
    )

    // val prediction: String
    val fakeAudioResponse = AudioResponse(prediction = "FAKE")

    // val claim: String,
    //    val result: String,
    //    val similarity_score: Double,
    //    val sources: List<Source>
    val listOfSource = listOf<Source>(
        Source(title = "XYZ","www.xyz.com"),
        Source(title = "ABC","www.abcd.com")

    )
    val fakeNewResponse = NewResponse(
        claim = "Donald Trump is president of USA",
        result = "FAKE NEWS",
        similarity_score = 0.5,
        sources = listOfSource
    )

    //Test
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

    override suspend fun getGradCamFromServer(): Flow<ApiResult<GradCamResponse>> =flow{

        emit(ApiResult.Success(fakeGradCamResponse))

    }

    override suspend fun uploadAudioToDeepFakeServer(
        context: Context,
        audioUrl: String
    ): Flow<ApiResult<AudioResponse>> =flow{
       emit(ApiResult.Success(fakeAudioResponse))
    }

    override suspend fun newsPrediction(claim: String): Flow<ApiResult<NewResponse>> =flow{

      emit(ApiResult.Success(fakeNewResponse))

    }

    override suspend fun imagePrediction(
        context: Context,
        imageUri: String
    ): Flow<ApiResult<ImageResponse>> {
        TODO("Not yet implemented")
    }
}