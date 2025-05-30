package com.example.deepshield.presentation.Screens

import androidx.compose.foundation.layout.Arrangement
import androidx.compose.foundation.layout.Row
import androidx.compose.foundation.layout.fillMaxSize
import androidx.compose.foundation.layout.fillMaxWidth
import androidx.compose.foundation.layout.padding
import androidx.compose.foundation.layout.widthIn
import androidx.compose.foundation.layout.wrapContentHeight
import androidx.compose.foundation.lazy.LazyColumn
import androidx.compose.foundation.lazy.items
import androidx.compose.foundation.shape.RoundedCornerShape
import androidx.compose.material3.Card
import androidx.compose.material3.CardDefaults
import androidx.compose.material3.Text
import androidx.compose.runtime.Composable
import androidx.compose.ui.Modifier
import androidx.compose.ui.graphics.Color
import androidx.compose.ui.unit.dp

@Composable
fun ChatScreen(messages: List<Message>) {
    LazyColumn(
        modifier = Modifier
            .fillMaxSize()
            .padding(8.dp)
    ) {
        items(messages) { message ->
            MessageItem(message)

        }
    }
}

@Composable
fun MessageItem(message: Message) {
    // Align messages based on who sent them
    val alignment = if (message.isUser) Arrangement.End else Arrangement.Start
    val backgroundColor = if (message.isUser) Color(0xFFDCF8C6) else Color(0xFFD0E8FF) // greenish for user, bluish for bot

    Row(
        modifier = Modifier
            .fillMaxWidth()
            .padding(vertical = 4.dp),
        horizontalArrangement = alignment
    ) {
        Card(
            modifier = Modifier
                .widthIn(max = 250.dp)
                .wrapContentHeight(),
            shape = RoundedCornerShape(12.dp),
            colors = CardDefaults.cardColors(containerColor = backgroundColor),
            elevation = CardDefaults.cardElevation(4.dp)
        ) {
            Text(
                text = message.text,
                modifier = Modifier.padding(12.dp),
                color = Color.Black
            )
        }
    }
}

// Simple data class for a message
data class Message(
    val text: String,
    val isUser: Boolean
)
@Composable
fun ChatBotMainScreen() {
    val sampleMessages = listOf(
        Message("Hey chatbot!", isUser = true),
        Message("Hello human! How can I help?", isUser = false),
        Message("Tell me a joke", isUser = true),
        Message("Why don’t scientists trust atoms? Because they make up everything!", isUser = false)
    )

    ChatScreen(messages = sampleMessages)
}
