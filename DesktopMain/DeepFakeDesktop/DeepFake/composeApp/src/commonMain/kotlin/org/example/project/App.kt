package org.example.project

import androidx.compose.foundation.layout.Column
import androidx.compose.foundation.layout.Spacer
import androidx.compose.foundation.layout.height
import androidx.compose.material3.Button
import androidx.compose.material3.MaterialTheme
import androidx.compose.material3.Text
import androidx.compose.runtime.*
import androidx.compose.ui.Modifier
import androidx.compose.ui.unit.dp
import androidx.navigation.NavController
import androidx.navigation.compose.NavHost
import androidx.navigation.compose.composable
import androidx.navigation.compose.rememberNavController
import com.example.deepshield.Constants.Constants
import io.github.vinceglb.filekit.PlatformFile
import io.github.vinceglb.filekit.dialogs.FileKitDialogSettings
import io.github.vinceglb.filekit.dialogs.FileKitType
import io.github.vinceglb.filekit.dialogs.compose.rememberFilePickerLauncher
import io.github.vinceglb.filekit.path
import io.github.vinceglb.filekit.readBytes
import io.ktor.client.HttpClient
import io.ktor.client.call.body
import io.ktor.client.engine.cio.CIO
import io.ktor.client.plugins.HttpTimeout
import io.ktor.client.plugins.contentnegotiation.ContentNegotiation
import io.ktor.client.request.forms.formData
import io.ktor.client.request.forms.submitFormWithBinaryData
import io.ktor.client.statement.HttpResponse
import io.ktor.http.Headers
import io.ktor.http.HttpHeaders
import io.ktor.serialization.kotlinx.json.json
import kotlinx.coroutines.GlobalScope
import kotlinx.coroutines.launch
import kotlinx.serialization.Serializable
import kotlinx.serialization.json.Json
import org.example.project.data.ApiResponse.DeepFakeVideoResponse
import org.jetbrains.compose.resources.painterResource
import org.jetbrains.compose.ui.tooling.preview.Preview


@Composable
@Preview
fun App() {
    MaterialTheme {
        var showContent by remember { mutableStateOf(false) }
        val navController= rememberNavController()
        NavHost(navController = navController, startDestination = SCREENA) {
            composable<SCREENA> {
                ScreenA(navController = navController)

            }
            composable<SCREENB> {
                ScreenB()

            }
        }


    }
}



@Composable
fun ScreenA(navController: NavController) {
    Text(text = "Hello Screen A!")
    Button(
        onClick = {
            navController.navigate(SCREENB)
        }
    ) {
     Text("Scrren B")
    }
}

@Composable
fun ScreenB() {
    val scope = rememberCoroutineScope()
    val httpClient = remember { provideHttpClient() }

    // 👉 State to hold the prediction result
    val predictionResult = remember { mutableStateOf<String?>(null) }

    val launcher = rememberFilePickerLauncher(
        title = "Select a video",
        type = FileKitType.File(extensions = listOf("mp4", "avi", "mkv", "mov", "webm")),
        dialogSettings = FileKitDialogSettings.createDefault()
    ) { file ->
        if (file != null) {
            println("Picked file: ${file.path}")

            scope.launch {
                val bytes = file.readBytes()
                try {
                    println("Uploading video...")
                    val response: HttpResponse = httpClient.submitFormWithBinaryData(
                        url = "${Constants.BASE_URL}${Constants.VIDEO_ROUTE}",
                        formData = formData {
                            append("video", bytes,
                                Headers.build {
                                    append(HttpHeaders.ContentType, "video/mp4")
                                    append(HttpHeaders.ContentDisposition, "form-data; name=\"video\"; filename=\"video.mp4\"")
                                }
                            )
                        }
                    )

                    val apiResponse: DeepFakeVideoResponse = response.body()
                    println("✅ Upload Successful: $apiResponse")

                    // ✅ Set the prediction result here
                    predictionResult.value = "Prediction: ${apiResponse.prediction}\nScore: ${apiResponse.score}"

                } catch (e: Exception) {
                    println("❌ Upload Failed: ${e.message}")
                    predictionResult.value = "Error: ${e.message}"
                }
            }
        } else {
            println("No file selected.")
        }
    }

    Column {
        Text(text = "Hello Screen B!")
        Spacer(modifier = Modifier.height(16.dp))
        Button(
            onClick = { launcher.launch() }
        ) {
            Text("Pick and Upload Video")
        }

        Spacer(modifier = Modifier.height(16.dp))

        // 👉 Show the prediction result below the button
        predictionResult.value?.let { result ->
            Text(text = result)
        }
    }
}

@Serializable
object SCREENA

@Serializable
object SCREENB

fun provideHttpClient():HttpClient{

    val client = HttpClient(CIO){
        install(HttpTimeout) {
            requestTimeoutMillis = 300_000 // 120 seconds (2 minutes)
            connectTimeoutMillis = 300_000  // 60 seconds (1 minute)
            socketTimeoutMillis = 300_000  // 120 seconds (2 minutes)
        }
        install(ContentNegotiation){
            json(
                Json {
                    ignoreUnknownKeys = true
                    isLenient=true
                    prettyPrint=true
                }
            )
        }

    }
    return client
}

