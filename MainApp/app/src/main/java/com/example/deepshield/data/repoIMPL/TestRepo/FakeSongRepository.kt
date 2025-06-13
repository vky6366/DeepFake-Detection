package com.example.deepshield.data.repoIMPL.TestRepo

import com.example.deepshield.data.Song.Song
import com.example.deepshield.domain.Repository.SongRepository
import com.example.deepshield.domain.StateHandling.ResultState
import kotlinx.coroutines.flow.Flow

class FakeSongRepository:SongRepository {
    override suspend fun getAllSongs(): Flow<ResultState<List<Song>>> {
        TODO("Not yet implemented")
    }
}