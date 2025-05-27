package com.example.deepshield.data.UseCases

import com.example.deepshield.data.Song.Song
import com.example.deepshield.domain.Repository.SongRepository
import com.example.deepshield.domain.StateHandling.ResultState
import kotlinx.coroutines.flow.Flow
import javax.inject.Inject

class GetAllSongUseCase @Inject constructor(private val repository: SongRepository) {
    operator suspend fun invoke(): Flow<ResultState<List<Song>>> {
        return repository.getAllSongs()
    }
}