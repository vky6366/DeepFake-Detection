package com.example.deepshield.data.UseCases

import android.content.Context
import com.example.deepshield.data.Response.AudioResponse
import com.example.deepshield.domain.Repository.Repository
import com.example.deepshield.domain.StateHandling.ApiResult
import kotlinx.coroutines.flow.Flow
import javax.inject.Inject

class UploadAudioToServerUseCase @Inject constructor(private val repository: Repository) {
    suspend operator fun invoke(context:Context,audioUrl:String): Flow<ApiResult<AudioResponse>> {
        return repository.uploadAudioToDeepFakeServer(context, audioUrl)
    }
}