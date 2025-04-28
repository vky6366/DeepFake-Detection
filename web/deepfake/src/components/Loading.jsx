// loading indicates that video is in processing state

function Loading (){
    return (
      <div className="mt-6 text-center">
        <div className="flex items-center justify-center gap-3">
          <div className="animate-spin rounded-full h-8 w-8 border-b-2 border-[#0D47A1]"></div>
          <span className="text-[#0D47A1] font-medium">Processing video... Please wait.</span>
        </div>
      </div>
    );
  };
  
  export default Loading;