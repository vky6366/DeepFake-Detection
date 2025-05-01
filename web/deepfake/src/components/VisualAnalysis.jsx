// visualanalysis result
function VisualAnalysis({ frameImg, gradcamImg }) {
    return (
        <section className="card p-8">
            <h2 className="text-3xl font-semibold text-center text-[#0D47A1]">Visual Analysis</h2>
            <div className="grid md:grid-cols-2 gap-8 mt-6">
                <div className="space-y-4 text-center">
                    <h3 className="font-semibold text-[#0D47A1]">Frame</h3>
                    <div className="relative aspect-video bg-[#E3F2FD] rounded-lg overflow-hidden">
                        {frameImg && <img src={frameImg} alt="Original Frame" className="w-full h-full object-contain" />}
                    </div>
                </div>

                <div className="space-y-4 text-center">
                    <h3 className="font-semibold text-[#0D47A1]">Heatmap</h3>
                    <div className="relative aspect-video bg-[#E3F2FD] rounded-lg overflow-hidden">
                        {gradcamImg && <img src={gradcamImg} alt="Grad-CAM Heatmap" className="w-full h-full object-contain" />}
                    </div>
                </div>
            </div>
        </section>
    );
};

export default VisualAnalysis;




