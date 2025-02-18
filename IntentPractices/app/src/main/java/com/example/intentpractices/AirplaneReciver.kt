package com.example.intentpractices

import android.content.BroadcastReceiver
import android.content.Context
import android.content.Intent

class AirplaneReciver:BroadcastReceiver() {
    override fun onReceive(context: Context?, intent: Intent?) {
        if(intent?.action== Intent.ACTION_AIRPLANE_MODE_CHANGED){
            val isEnabled = intent.getBooleanExtra("state", false)

        }
    }
}