package com.example.deepshield.domain.Repository

import com.example.deepshield.data.Song.Song
import com.example.deepshield.domain.StateHandling.ResultState
import kotlinx.coroutines.flow.Flow

interface SongRepository {
    suspend fun getAllSongs() :Flow<ResultState<List<Song>>>
}