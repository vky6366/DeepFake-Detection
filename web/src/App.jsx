// import React from 'react';
// import Header from './components/Header';
// import Display from './components/Display';
// import Loading from './components/Loading';
// import Result from './components/Result';
// import Upload from './components/Upload';
// import VisualAnalysis from './components/VisualAnalysis';

// const App=()=>{
//   return(
//     <div className='gradient-bg min-h-screen flex items-center justify-center'>
//       <Header />
//       <main className='container mx-auto px-6 py-12 flex flex-col gap-8 w-full max-w-3xl'>
//       <Display />
//       <Loading />
//       <Result />
//       <Upload />
//       <VisualAnalysis />
//       </main>
//     </div>
//   )
// }

// export default App;

import React from 'react';
import Display from './components/Display';

const App = () => {
  return (
    <>
      <div className='gradient-bg min-h-screen flex items-center justify-center'>
        <Display />
      </div>
    </>

  );
};

export default App;
