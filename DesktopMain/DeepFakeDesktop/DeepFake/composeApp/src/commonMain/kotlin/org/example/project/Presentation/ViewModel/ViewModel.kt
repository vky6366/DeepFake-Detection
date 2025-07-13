package org.example.project.Presentation.ViewModel

import androidx.lifecycle.ViewModel
import androidx.lifecycle.viewModelScope
import kotlinx.coroutines.flow.MutableStateFlow
import kotlinx.coroutines.flow.asStateFlow
import kotlinx.coroutines.flow.collectLatest
import kotlinx.coroutines.launch
import org.example.project.data.stateHandler.ApiResult
import org.example.project.data.stateHandler.AudioResponseState
import org.example.project.data.stateHandler.DeepFakeVideoResponseState
import org.example.project.data.stateHandler.ImageResponseState
import org.example.project.data.stateHandler.NewsPredictionState
import org.example.project.domain.UseCase.NewsPredictionUseCase
import org.example.project.domain.UseCase.UploadAudioToDeepFakeServerUseCase
import org.example.project.domain.UseCase.UploadImageUseCase
import org.example.project.domain.UseCase.UploadVideoToDeepFakeServerUseCase


class MyViewModel(private val uploadVideoToDeepFakeServerUseCase: UploadVideoToDeepFakeServerUseCase,
                  private val uploadAudioToDeepFakeServerUseCase: UploadAudioToDeepFakeServerUseCase,
                  private val uploadImageUseCase: UploadImageUseCase,
                  private val newsPredictionUseCase: NewsPredictionUseCase
): ViewModel() {

    private val _uploadDeepFakeVideoState = MutableStateFlow(DeepFakeVideoResponseState())
    val uploadDeepFakeVideoState = _uploadDeepFakeVideoState.asStateFlow()

    private val _uploadAudioToServerState = MutableStateFlow(AudioResponseState())

    val uploadAudioToServerState = _uploadAudioToServerState.asStateFlow()

    private val _imagePredictionState = MutableStateFlow(ImageResponseState())

    val imagePredictionState = _imagePredictionState.asStateFlow()

    private val _newPredictionState = MutableStateFlow(NewsPredictionState())

    val newPredictionState = _newPredictionState.asStateFlow()

    // same for others...

    fun uploadVideoToDeepFakeServer(videoBytes: ByteArray) {
        viewModelScope.launch {
            uploadVideoToDeepFakeServerUseCase.invoke(videoBytes).collectLatest {
                when (it) {
                    is ApiResult.Loading -> {
                        _uploadDeepFakeVideoState.value = DeepFakeVideoResponseState(isLoading = true)
                    }
                    is ApiResult.Error -> {
                        _uploadDeepFakeVideoState.value = DeepFakeVideoResponseState(isLoading = false, error = it.message)
                    }
                    is ApiResult.Success -> {
                        _uploadDeepFakeVideoState.value = DeepFakeVideoResponseState(isLoading = false, data = it.data)
                    }
                }
            }
        }
    }

    fun uploadAudioToDeepFakeServer(audioUrl: ByteArray) {
        viewModelScope.launch {
            uploadAudioToDeepFakeServerUseCase.invoke(audioUrl).collectLatest { result ->
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

    fun imagePrediction(imageUri: ByteArray){
        viewModelScope.launch {
           uploadImageUseCase.invoke(imageUri=imageUri).collectLatest {result->
                when(result){
                    is ApiResult.Loading->{
                        _imagePredictionState.value =ImageResponseState(isLoading = true)
                    }
                    is ApiResult.Error->{
                        _imagePredictionState.value = ImageResponseState(isLoading = false, error = result.message)
                    }
                    is ApiResult.Success->{
                        _imagePredictionState.value = ImageResponseState(isLoading = false, data = result.data)
                    }
                }

            }

        }
    }

    fun newPrediction(claim: String) {
        viewModelScope.launch {
            newsPredictionUseCase.invoke(claim = claim).collectLatest {result->
                when(result){
                    is ApiResult.Loading->{
                        _newPredictionState.value = NewsPredictionState(isLoading = true)
                    }
                    is ApiResult.Error->{
                        _newPredictionState.value = NewsPredictionState(isLoading = false, error = result.message)
                    }
                    is ApiResult.Success->{
                        _newPredictionState.value = NewsPredictionState(isLoading = false, data = result.data)
                    }
                }

            }
        }
    }


}
