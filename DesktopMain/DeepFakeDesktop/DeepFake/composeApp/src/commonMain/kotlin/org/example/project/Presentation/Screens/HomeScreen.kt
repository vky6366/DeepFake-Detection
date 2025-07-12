package org.example.project.Presentation.Screens

import androidx.compose.foundation.background
import androidx.compose.foundation.border
import androidx.compose.foundation.clickable
import androidx.compose.foundation.layout.*
import androidx.compose.foundation.shape.CircleShape
import androidx.compose.foundation.shape.RoundedCornerShape
import androidx.compose.material3.*
import androidx.compose.runtime.Composable
import androidx.compose.ui.Alignment
import androidx.compose.ui.Modifier
import androidx.compose.ui.draw.clip
import androidx.compose.ui.draw.shadow
import androidx.compose.ui.graphics.Color
import androidx.compose.ui.text.font.FontWeight
import androidx.compose.ui.unit.dp
import androidx.compose.ui.unit.sp
import androidx.navigation.NavController
import org.example.project.Presentation.Navigation.AUDIOUPLOADSCREEN
import org.example.project.Presentation.Navigation.IMAGEUPLOADSCREEN
import org.example.project.Presentation.Navigation.VIDEOUPLOADSCREEN


@Composable
fun HomeScreenUI(navController: NavController) {
    MaterialTheme {
        Box(
            modifier = Modifier
                .fillMaxSize()
                .background(Color(0xFFF8F6FB)),
            contentAlignment = Alignment.Center
        ) {
            Column(
                horizontalAlignment = Alignment.CenterHorizontally,
                verticalArrangement = Arrangement.spacedBy(32.dp),
            ) {
                Text(
                    text = "🛡️ Deep Shield",
                    style = MaterialTheme.typography.headlineLarge.copy(
                        fontSize = 36.sp,
                        fontWeight = FontWeight.Bold,
                        color = Color.Black
                    )
                )

                Column(
                    verticalArrangement = Arrangement.spacedBy(20.dp),
                    horizontalAlignment = Alignment.CenterHorizontally
                ) {
                    Row(horizontalArrangement = Arrangement.spacedBy(20.dp)) {
                        DashboardCard("Video", Color(0xFFE0EDFF), "🎥") {
                           navController.navigate(VIDEOUPLOADSCREEN)
                        }
                        DashboardCard("Image", Color(0xFFFFDDE4), "🖼️") {
                           navController.navigate(IMAGEUPLOADSCREEN)
                        }
                    }
                    Row(horizontalArrangement = Arrangement.spacedBy(20.dp)) {
                        DashboardCard("Audio", Color(0xFFE6FFE7), "🎵") {
                           navController.navigate(AUDIOUPLOADSCREEN)
                        }
                        DashboardCard("News", Color(0xFFE1F5FE), "📰") {
                            println("📰 News Clicked")
                        }
                    }
                }
            }
        }
    }
}
@Composable
fun DashboardCard(
    label: String,
    badgeColor: Color,
    emoji: String,
    onClick: () -> Unit // 💥 Now accepts a click action
) {
    Box(
        modifier = Modifier
            .size(160.dp)
            .clip(RoundedCornerShape(24.dp))
            .background(Color.White)
            .border(2.dp, Color.LightGray, RoundedCornerShape(24.dp))
            .clickable { onClick() } // 👈 use it here
            .shadow(8.dp, RoundedCornerShape(24.dp)),
        contentAlignment = Alignment.Center
    ) {
        Column(
            horizontalAlignment = Alignment.CenterHorizontally,
            verticalArrangement = Arrangement.spacedBy(12.dp)
        ) {
            Box(
                modifier = Modifier
                    .size(48.dp)
                    .clip(CircleShape)
                    .background(badgeColor),
                contentAlignment = Alignment.Center
            ) {
                Text(
                    text = emoji,
                    fontSize = 28.sp
                )
            }

            Text(
                text = label,
                style = MaterialTheme.typography.bodyLarge.copy(
                    fontWeight = FontWeight.SemiBold,
                    fontSize = 18.sp
                )
            )
        }
    }
}
