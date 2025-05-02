import Header from "./components/Header";
import Loading from "./components/Loading";
import Result from "./components/Result";
import Upload from "./components/Upload";
import VisualAnalysis from "./components/VisualAnalysis";
import api from "./server/api";

function App(){
  return(
    <>
      <Header />
      <main className="gradient-bg min-h-screen flex flex-col items-center">
        <div className="container mx-auto px-6 py-12 flex flex-col gap-8 w-full max-w-3xl">
          <Upload />
          <Loading />
          <Result />
          <VisualAnalysis />

        </div>
      </main>
    </>
  )
}