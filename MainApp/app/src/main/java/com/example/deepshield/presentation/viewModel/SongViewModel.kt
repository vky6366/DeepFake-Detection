package com.example.deepshield.presentation.viewModel

import androidx.lifecycle.ViewModel
import androidx.lifecycle.viewModelScope
import com.example.deepshield.data.UseCases.UseCaseHelper.UseCaseHelperClass
import com.example.deepshield.domain.StateHandling.GetAllSongState
import com.example.deepshield.domain.StateHandling.ResultState
import dagger.hilt.android.lifecycle.HiltViewModel
import kotlinx.coroutines.Dispatchers
import kotlinx.coroutines.flow.MutableStateFlow
import kotlinx.coroutines.flow.asStateFlow
import kotlinx.coroutines.launch
import javax.inject.Inject

@HiltViewModel
class SongViewModel @Inject constructor(private val usecase:UseCaseHelperClass): ViewModel() {
    private val _getAllSongsState= MutableStateFlow(GetAllSongState())
    val getAllSongsState = _getAllSongsState.asStateFlow()
    init {
        getAllSong()
    }
    fun getAllSong(){
        viewModelScope.launch(Dispatchers.IO) {
            usecase.getAllSongUseCase.invoke().collect {result->
                when(result){
                    is ResultState.Loading->{
                        _getAllSongsState.value = GetAllSongState(isLoading = true)
                    }
                    is ResultState.Success ->{
                        _getAllSongsState.value = GetAllSongState(isLoading = false, data = result.data)
                    }
                    is ResultState.Error->{
                        _getAllSongsState.value = GetAllSongState(isLoading = false, error = result.message)

                    }
                }

            }
        }
    }


}