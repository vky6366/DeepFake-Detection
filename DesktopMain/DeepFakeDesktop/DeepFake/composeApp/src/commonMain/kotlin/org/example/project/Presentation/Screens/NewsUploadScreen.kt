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
import androidx.compose.material3.Divider
import androidx.compose.material3.MaterialTheme
import androidx.compose.material3.OutlinedTextField
import androidx.compose.material3.Text
import androidx.compose.runtime.Composable
import androidx.compose.runtime.collectAsState
import androidx.compose.runtime.getValue
import androidx.compose.runtime.mutableStateOf
import androidx.compose.runtime.remember
import androidx.compose.runtime.rememberCoroutineScope
import androidx.compose.runtime.setValue
import androidx.compose.ui.Alignment
import androidx.compose.ui.Modifier
import androidx.compose.ui.graphics.Color
import androidx.compose.ui.text.font.FontWeight
import androidx.compose.ui.unit.dp
import androidx.compose.ui.unit.sp
import org.example.project.Presentation.ViewModel.MyViewModel

@Composable
fun NewsCheckScreen(
    viewModel: MyViewModel
) {
    val scope = rememberCoroutineScope()
    val newsState by viewModel.newPredictionState.collectAsState()

    var claimText by remember { mutableStateOf("") }

    Box(
        modifier = Modifier
            .fillMaxSize()
            .background(Color(0xFFF9FAFC))
            .padding(32.dp),
        contentAlignment = Alignment.Center
    ) {
        Card(
            modifier = Modifier.widthIn(min = 400.dp, max = 700.dp),
            shape = RoundedCornerShape(20.dp),
            colors = CardDefaults.cardColors(containerColor = Color.White),
            elevation = CardDefaults.cardElevation(8.dp)
        ) {
            Column(
                modifier = Modifier.padding(32.dp),
                verticalArrangement = Arrangement.spacedBy(20.dp),
                horizontalAlignment = Alignment.CenterHorizontally
            ) {

                // 📰 Title
                Text(
                    text = "News Claim Verifier",
                    style = MaterialTheme.typography.headlineMedium.copy(
                        fontWeight = FontWeight.Bold,
                        color = Color(0xFF1A1A1A)
                    )
                )

                // 📝 Text Field
                OutlinedTextField(
                    value = claimText,
                    onValueChange = { claimText = it },
                    label = { Text("Enter a news claim...") },
                    modifier = Modifier.fillMaxWidth(),
                    shape = RoundedCornerShape(16.dp),
                    maxLines = 4
                )

                // 🔍 Button
                Button(
                    onClick = {
                        if (claimText.isNotBlank()) {
                            viewModel.newPrediction(claimText)
                        }
                    },
                    shape = RoundedCornerShape(16.dp),
                    colors = ButtonDefaults.buttonColors(containerColor = Color(0xFF4C6EF5))
                ) {
                    Text("🧠 Verify", color = Color.White)
                }

                // 🌀 Loading
                if (newsState.isLoading) {
                    CircularProgressIndicator(
                        color = Color(0xFF4C6EF5),
                        strokeWidth = 4.dp
                    )
                }

                // ❌ Error
                if (!newsState.error.isNullOrBlank()) {
                    Text(
                        text = "❌ Error: ${newsState.error}",
                        color = MaterialTheme.colorScheme.error,
                        modifier = Modifier
                            .fillMaxWidth()
                            .background(Color(0xFFFFEBEE), shape = RoundedCornerShape(8.dp))
                            .padding(12.dp)
                    )
                }

                // ✅ Result UI
                newsState.data?.let { data ->
                    Column(
                        modifier = Modifier
                            .fillMaxWidth()
                            .background(Color(0xFFE3F2FD), shape = RoundedCornerShape(12.dp))
                            .padding(16.dp),
                        verticalArrangement = Arrangement.spacedBy(8.dp)
                    ) {
                        Text(
                            text = "📝 Claim: ${data.claim}",
                            style = MaterialTheme.typography.bodyLarge.copy(fontWeight = FontWeight.SemiBold),
                            color = Color(0xFF0D47A1)
                        )
                        Text(
                            text = "✅ Result: ${data.result}",
                            style = MaterialTheme.typography.bodyLarge,
                            color = if (data.result.lowercase() == "true") Color(0xFF2E7D32) else Color(0xFFD32F2F)
                        )
                        Text(
                            text = "🎯 Similarity Score: ${data.similarity_score}",
                            style = MaterialTheme.typography.bodyLarge,
                            color = Color(0xFF1E88E5)
                        )

                        Divider(color = Color.Gray, thickness = 1.dp)

                        Text(
                            text = "📚 Sources:",
                            style = MaterialTheme.typography.titleMedium.copy(fontWeight = FontWeight.Bold)
                        )

                        data.sources.forEach { source ->
                            Column(modifier = Modifier.padding(bottom = 8.dp)) {
                                Text(
                                    text = "🔗 ${source.title}",
                                    color = Color(0xFF0D47A1),
                                    fontWeight = FontWeight.Medium
                                )
                                Text(
                                    text = source.url,
                                    color = Color(0xFF2962FF),
                                    fontSize = 13.sp
                                )
                            }
                        }
                    }
                }
            }
        }
    }
}
