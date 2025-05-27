package com.example.deepshield.presentation.Screens

import androidx.compose.foundation.background
import androidx.compose.foundation.clickable
import androidx.compose.foundation.layout.Arrangement
import androidx.compose.foundation.layout.Box
import androidx.compose.foundation.layout.Column
import androidx.compose.foundation.layout.Row
import androidx.compose.foundation.layout.fillMaxSize
import androidx.compose.foundation.layout.wrapContentSize
import androidx.compose.foundation.lazy.LazyColumn
import androidx.compose.foundation.lazy.items
import androidx.compose.material.icons.Icons
import androidx.compose.material.icons.filled.Search
import androidx.compose.material3.CircularProgressIndicator
import androidx.compose.material3.Icon
import androidx.compose.material3.IconButton
import androidx.compose.material3.MaterialTheme
import androidx.compose.material3.OutlinedTextField
import androidx.compose.material3.OutlinedTextFieldDefaults
import androidx.compose.material3.Text
import androidx.compose.runtime.Composable
import androidx.compose.runtime.LaunchedEffect
import androidx.compose.runtime.collectAsState
import androidx.compose.ui.Modifier
import androidx.compose.ui.graphics.Color
import androidx.compose.ui.text.TextStyle
import androidx.core.net.toUri
import androidx.hilt.navigation.compose.hiltViewModel
import androidx.navigation.NavController
import com.example.deepshield.presentation.viewModel.SongViewModel

@Composable
fun AllSongScreen(viewmodel: SongViewModel = hiltViewModel(),navController: NavController) {
    LaunchedEffect(Unit) {
        viewmodel.getAllSong()
    }
    val songState = viewmodel.getAllSongsState.collectAsState()
    when{
        songState.value.isLoading -> {
            CircularProgressIndicator()
        }
        !songState.value.error.isNullOrEmpty() -> {
            Text(text = songState.value.error.toString())
        }
        !songState.value.data.isNullOrEmpty()->{
            Column(modifier = Modifier.fillMaxSize()) {

                LazyColumn(
                    modifier = Modifier
                        .weight(0.90f)
                ) {
                    items(songState.value.data) { song ->
                        Text(song.title.toString(), color = Color.Red)
                    }

                }
            }

        }
    }

}