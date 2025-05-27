package com.example.deepshield.di

import android.content.Context
import com.example.deepshield.data.UseCases.GetAllSongUseCase
import com.example.deepshield.data.UseCases.GetFrameFromServerUseCase
import com.example.deepshield.data.UseCases.GetGradCamUseCase
import com.example.deepshield.data.UseCases.GetHeatMapUseCase
import com.example.deepshield.data.UseCases.UploadVideoToServerUseCase
import com.example.deepshield.data.UseCases.UseCaseHelper.UseCaseHelperClass
import com.example.deepshield.data.repoIMPL.RepositoryImpl
import com.example.deepshield.data.repoIMPL.SongRepoImpl
import com.example.deepshield.domain.Repository.Repository
import com.example.deepshield.domain.Repository.SongRepository
import dagger.Module
import dagger.Provides
import dagger.hilt.InstallIn
import dagger.hilt.android.qualifiers.ApplicationContext
import dagger.hilt.components.SingletonComponent
import javax.inject.Singleton

@Module
@InstallIn(SingletonComponent::class)
object DiObject {
    @Provides
    fun provideRepoIMPL():RepositoryImpl{
        return RepositoryImpl()
    }
    @Provides
    fun provideRepository():Repository{
        return RepositoryImpl()
    }
    @Provides
    fun getAllSongUseCaseObj(songRepository: SongRepository): GetAllSongUseCase{
        return GetAllSongUseCase(repository = songRepository)
    }
    @Provides
    fun useCaseHelperClass(@ApplicationContext context: Context):UseCaseHelperClass{
        return UseCaseHelperClass(
            getGradCamFromServerUseCase = GetGradCamUseCase(repository = provideRepository()),
            getFrameFromServerUseCase = GetFrameFromServerUseCase(repository = provideRepository()),
            getHeatMapFromServerUseCase = GetHeatMapUseCase(repository = provideRepository()),
            uploadVideoToDeepFakeServerUseCase = UploadVideoToServerUseCase(repository = provideRepository()),
            getAllSongUseCase=getAllSongUseCaseObj(songRepository = providesongRepoObj(context = context))
        )
    }
    @Provides
    fun providesongRepoObj(@ApplicationContext context: Context): SongRepository{
        return SongRepoImpl(context =context )
    }

}