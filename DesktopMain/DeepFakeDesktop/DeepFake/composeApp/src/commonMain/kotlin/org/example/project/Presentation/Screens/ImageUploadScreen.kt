package org.example.project.Presentation.Screens

import androidx.compose.foundation.background
import androidx.compose.foundation.layout.Arrangement
import androidx.compose.foundation.layout.Box
import androidx.compose.foundation.layout.Column
import androidx.compose.foundation.layout.fillMaxSize
import androidx.compose.foundation.layout.fillMaxWidth
import androidx.compose.foundation.layout.padding
import androidx.compose.foundation.layout.widthIn
import androidx.compose.foundation.shape.RoundedCornerShape
import androidx.compose.material3.Button
import androidx.compose.material3.ButtonDefaults
import androidx.compose.material3.Card
import androidx.compose.material3.CardDefaults
import androidx.compose.material3.CircularProgressIndicator
import androidx.compose.material3.MaterialTheme
import androidx.compose.material3.Text
import androidx.compose.runtime.Composable
import androidx.compose.runtime.collectAsState
import androidx.compose.runtime.getValue
import androidx.compose.runtime.rememberCoroutineScope
import androidx.compose.ui.Alignment
import androidx.compose.ui.Modifier
import androidx.compose.ui.graphics.Color
import androidx.compose.ui.text.font.FontWeight
import androidx.compose.ui.unit.dp
import io.github.vinceglb.filekit.dialogs.FileKitDialogSettings
import io.github.vinceglb.filekit.dialogs.FileKitType
import io.github.vinceglb.filekit.dialogs.compose.rememberFilePickerLauncher
import io.github.vinceglb.filekit.readBytes
import kotlinx.coroutines.launch
import org.example.project.Presentation.ViewModel.MyViewModel

@Composable
fun ImageUploadScreen(
    viewModel: MyViewModel
) {
    val scope = rememberCoroutineScope()

    // 🧠 Observe image upload state
    val uploadState by viewModel.imagePredictionState.collectAsState()

    // 📁 File picker for image
    val launcher = rememberFilePickerLauncher(
        title = "Pick an image file",
        type = FileKitType.Image, // Accepts png, jpg, etc.
        dialogSettings = FileKitDialogSettings.createDefault()
    ) { file ->
        file?.let {
            scope.launch {
                val imageBytes = it.readBytes()
                viewModel.imagePrediction(imageBytes)
            }
        }
    }

    // 🎨 UI Layout (same as AudioUploadScreen)
    Box(
        modifier = Modifier
            .fillMaxSize()
            .background(Color(0xFFF4F6FA))
            .padding(32.dp),
        contentAlignment = Alignment.Center
    ) {
        Card(
            modifier = Modifier.widthIn(min = 400.dp, max = 600.dp),
            shape = RoundedCornerShape(24.dp),
            colors = CardDefaults.cardColors(containerColor = Color.White),
            elevation = CardDefaults.cardElevation(8.dp)
        ) {
            Column(
                modifier = Modifier.padding(32.dp),
                verticalArrangement = Arrangement.spacedBy(24.dp),
                horizontalAlignment = Alignment.CenterHorizontally
            ) {
                // 🖼️ Title
                Text(
                    text = "Upload Image for DeepFake Detection",
                    style = MaterialTheme.typography.headlineSmall.copy(
                        fontWeight = FontWeight.Bold,
                        color = Color(0xFF1A1A1A)
                    )
                )

                // 📷 Upload Button
                Button(
                    onClick = { launcher.launch() },
                    shape = RoundedCornerShape(16.dp),
                    colors = ButtonDefaults.buttonColors(containerColor = Color(0xFF4C6EF5))
                ) {
                    Text(
                        text = "🖼️ Pick and Upload",
                        color = Color.White,
                        style = MaterialTheme.typography.bodyLarge
                    )
                }

                // 🌀 Loading
                if (uploadState.isLoading) {
                    CircularProgressIndicator(
                        color = Color(0xFF4C6EF5),
                        strokeWidth = 4.dp
                    )
                }

                // ❌ Error
                if (!uploadState.error.isNullOrBlank()) {
                    Text(
                        text = "❌ Error: ${uploadState.error}",
                        color = MaterialTheme.colorScheme.error,
                        style = MaterialTheme.typography.bodyMedium,
                        modifier = Modifier
                            .background(Color(0xFFFFEBEE), shape = RoundedCornerShape(8.dp))
                            .padding(12.dp)
                    )
                }

                // ✅ Result
                uploadState.data?.let { data ->
                    Column(
                        modifier = Modifier
                            .fillMaxWidth()
                            .background(Color(0xFFE3F2FD), shape = RoundedCornerShape(12.dp))
                            .padding(16.dp),
                        verticalArrangement = Arrangement.spacedBy(8.dp)
                    ) {
                        Text(
                            text = "✅ Prediction: ${data.result}",
                            style = MaterialTheme.typography.bodyLarge.copy(fontWeight = FontWeight.Medium),
                            color = Color(0xFF0D47A1)
                        )
                        Text(
                            text = "🎯 Score: ${data.message}",
                            style = MaterialTheme.typography.bodyLarge.copy(fontWeight = FontWeight.Medium),
                            color = Color(0xFF0D47A1)
                        )
                    }
                }
            }
        }
    }
}
