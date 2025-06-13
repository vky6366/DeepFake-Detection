package com.example.deepshield.data.repoIMPL.TestRepo

import com.example.deepshield.data.Song.Song
import com.example.deepshield.domain.Repository.SongRepository
import com.example.deepshield.domain.StateHandling.ResultState
import kotlinx.coroutines.flow.Flow
import kotlinx.coroutines.flow.flow

class FakeSongRepository:SongRepository {
    // Example Song instance with placeholder values
    val fakeSong1 = Song(
        id = "123",
        path = "/storage/emulated/0/Music/song1.mp3",
        size = "5242880", // e.g., 5 MB in bytes
        album = "Greatest Hits",
        title = "Awesome Song",
        artist = "The Testers",
        duration = "180000", // e.g., 3 minutes in milliseconds
        year = "2023",
        composer = "John Doe",
        albumId = "album_456",
        albumArt = null // You might create a dummy Bitmap if needed, see Option 2
    )

    // You can create a list if your repository deals with multiple songs
    val fakeSongsList = listOf(
        fakeSong1,
        Song(
            id = "124",
            path = "/storage/emulated/0/Music/another_song.mp3",
            size = "4194304", // 4 MB
            album = "Rock Anthems",
            title = "Another Cool Track",
            artist = "The Mockers",
            duration = "240000", // 4 minutes
            year = "2022",
            composer = "Jane Smith",
            albumId = "album_789",
            albumArt = null
        )
    )
    override suspend fun getAllSongs(): Flow<ResultState<List<Song>>> = flow{
       emit(ResultState.Success(fakeSongsList))
    }
}