package com.example.deepshield.presentation.viewModel

import android.content.Context
import android.util.Log
import androidx.lifecycle.ViewModel
import androidx.lifecycle.viewModelScope
import com.example.deepshield.data.UseCases.UseCaseHelper.UseCaseHelperClass
import com.example.deepshield.data.repoIMPL.RepositoryImpl
import com.example.deepshield.domain.StateHandling.ApiResult
import com.example.deepshield.domain.StateHandling.AudioResponseState
import com.example.deepshield.domain.StateHandling.DeepFakeVideoResponseState
import com.example.deepshield.domain.StateHandling.FrameResponseState
import com.example.deepshield.domain.StateHandling.GradCamResponseState
import com.example.deepshield.domain.StateHandling.HeatMapResponseState
import dagger.hilt.android.lifecycle.HiltViewModel
import kotlinx.coroutines.flow.MutableStateFlow
import kotlinx.coroutines.flow.asStateFlow
import kotlinx.coroutines.flow.collectLatest
import kotlinx.coroutines.launch
import javax.inject.Inject

@HiltViewModel
class MyViewModel @Inject constructor(private val usecase:UseCaseHelperClass):ViewModel() {
    private val _uploadDeepFakeVideoState = MutableStateFlow(DeepFakeVideoResponseState())
    val uploadDeepFakeVideoState = _uploadDeepFakeVideoState.asStateFlow()
    private val _uploadAudioToServerState = MutableStateFlow(AudioResponseState())
    val uploadAudioToServerState = _uploadAudioToServerState.asStateFlow()

    // ✅ State for fetching a frame
    private val _frameState = MutableStateFlow(FrameResponseState())
    val frameState = _frameState.asStateFlow()

    private val _getHeatMapFromServerState = MutableStateFlow(HeatMapResponseState())
    val getHeatMapFromServerState = _getHeatMapFromServerState.asStateFlow()

    private val _getGradCamFromServerState = MutableStateFlow(GradCamResponseState())
    val getGradCamFromServerState = _getGradCamFromServerState.asStateFlow()

    fun uploadVideoToDeepFakeServer(context: Context, videoUri: String) {
        viewModelScope.launch {
            usecase.uploadVideoToDeepFakeServerUseCase.execute(
                context = context,
                videoUrl = videoUri
            )
                .collectLatest {
                    when (it) {
                        is ApiResult.Loading -> {
                            _uploadDeepFakeVideoState.value =
                                DeepFakeVideoResponseState(isLoading = true)

                        }

                        is ApiResult.Error -> {
                            _uploadDeepFakeVideoState.value =
                                DeepFakeVideoResponseState(isLoading = false, error = it.message)
                        }

                        is ApiResult.Success -> {
                            _uploadDeepFakeVideoState.value =
                                DeepFakeVideoResponseState(isLoading = false, data = it.data)

                        }
                    }
                }
        }

    }

    fun getFrameFromServer() {
        viewModelScope.launch {
            usecase.getFrameFromServerUseCase.execute().collectLatest { result ->
                when (result) {
                    is ApiResult.Loading -> {
                        _frameState.value = FrameResponseState(isLoading = true)
                    }

                    is ApiResult.Error -> {
                        _frameState.value =
                            FrameResponseState(isLoading = false, error = result.message)
                    }

                    is ApiResult.Success -> {
                        _frameState.value =
                            FrameResponseState(isLoading = false, bitmap = result.data)
                    }
                }
            }
        }
    }

    fun getHeatMapFromServer() {
        viewModelScope.launch {
            usecase.getHeatMapFromServerUseCase.execute().collect { response ->
                when (response) {
                    is ApiResult.Loading -> {
                        _getHeatMapFromServerState.value = HeatMapResponseState(isLoading = true)
                    }

                    is ApiResult.Error -> {
                        _getHeatMapFromServerState.value =
                            HeatMapResponseState(isLoading = false, error = response.message)
                    }

                    is ApiResult.Success -> {
                        _getHeatMapFromServerState.value =
                            HeatMapResponseState(isLoading = false, data = response.data)
                    }
                }

            }

        }
    }

    fun getGradCamResponse() {
        viewModelScope.launch {
            usecase.getGradCamFromServerUseCase.execute().collectLatest { result ->
                when (result) {
                    is ApiResult.Loading -> {
                        _getGradCamFromServerState.value = GradCamResponseState(isLoading = true)
                    }

                    is ApiResult.Success -> {
                        _getGradCamFromServerState.value =
                            GradCamResponseState(isLoading = false, data = result.data)
                    }

                    is ApiResult.Error -> {
                        _getGradCamFromServerState.value =
                            GradCamResponseState(isLoading = false, error = result.message)
                    }
                }

            }
        }

    }

    fun uploadAudioToDeepFakeServer(context: Context, audioUrl: String) {
        viewModelScope.launch {
            usecase.uploadAudioToServerUseCase.invoke(context,audioUrl).collectLatest { result ->
                when (result) {
                    is ApiResult.Loading -> {
                        _uploadAudioToServerState.value = AudioResponseState(isLoading = true)
                    }

                    is ApiResult.Error -> {
                        _uploadAudioToServerState.value =
                            AudioResponseState(isLoading = false, error = result.message)
                    }

                    is ApiResult.Success -> {
                        _uploadAudioToServerState.value = AudioResponseState(isLoading = false, data = result.data)
                    }
                }
            }
        }
    }


}


