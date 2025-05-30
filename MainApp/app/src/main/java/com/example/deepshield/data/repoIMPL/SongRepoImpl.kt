package com.example.deepshield.data.repoIMPL

import android.content.Context
import android.provider.MediaStore
import com.example.deepshield.data.Song.Song
import com.example.deepshield.domain.Repository.SongRepository
import com.example.deepshield.domain.StateHandling.ResultState
import kotlinx.coroutines.flow.Flow
import kotlinx.coroutines.flow.flow
import javax.inject.Inject

class SongRepoImpl @Inject constructor(private val context: Context): SongRepository {
    override suspend fun getAllSongs(): Flow<ResultState<List<Song>>> =flow{
        val songs = mutableListOf<Song>()
        val uri = MediaStore.Audio.Media.EXTERNAL_CONTENT_URI
        val projection = arrayOf(
            MediaStore.Audio.Media._ID,
            MediaStore.Audio.Media.DATA,
            MediaStore.Audio.Media.SIZE,
            MediaStore.Audio.Media.ALBUM,
            MediaStore.Audio.Media.TITLE,
            MediaStore.Audio.Media.ARTIST,
            MediaStore.Audio.Media.DURATION,
            MediaStore.Audio.Media.YEAR,
            MediaStore.Audio.Media.COMPOSER,
            MediaStore.Audio.Media.ALBUM_ID

        )
        val contentResolver = context.contentResolver
        val selection = "${MediaStore.Audio.Media.IS_MUSIC}!=0"
        emit(ResultState.Loading)
        try {
            val cursor = contentResolver.query(uri,projection,null,null,null)
            cursor?.use {cursorelement->
                while (cursorelement.moveToNext()){
                    val id = cursorelement.getString(0)
                    val path = cursorelement.getString(1)
                    val size = cursorelement.getString(2)
                    val album = cursorelement.getString(3)
                    val title = cursorelement.getString(4)
                    val artist = cursorelement.getString(5)
                    val duration = cursorelement.getString(6)
                    val year = cursorelement.getString(7)
                    val composer = cursorelement.getString(8)
                    val albumID = cursorelement.getString(9)
                    val song = Song(
                        id=id,
                        path = path,
                        size = size,
                        album = album,
                        title = title,
                        artist = artist,
                        duration = duration,
                        year = year,
                        composer = composer,
                        albumId = albumID
                    )
                    songs.add(song)
                }
            }
            emit(ResultState.Success(data = songs))

        }catch (e: Exception){
            emit(ResultState.Error(message = e.message.toString()))

        }
    }

}