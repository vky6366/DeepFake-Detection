package com.example.deepshield.data.KtorClient

import io.ktor.client.HttpClient
import io.ktor.client.engine.cio.CIO
import io.ktor.client.plugins.contentnegotiation.ContentNegotiation
import io.ktor.serialization.kotlinx.json.json
import kotlinx.serialization.json.Json

//Ktor Client Setup
object KtorClient {
    val client = HttpClient(CIO){
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
}