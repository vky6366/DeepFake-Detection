package com.example.deepshield.di

import com.example.deepshield.data.repoIMPL.RepositoryImpl
import dagger.Module
import dagger.Provides
import dagger.hilt.InstallIn
import dagger.hilt.components.SingletonComponent
import javax.inject.Singleton

@Module
@InstallIn(SingletonComponent::class)
object DiObject {
    @Provides
    fun provideRepoIMPL():RepositoryImpl{
        return RepositoryImpl()
    }

}