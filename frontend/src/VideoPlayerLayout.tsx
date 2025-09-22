import React, { useState, useEffect } from 'react';

type VideoSourceType = 'url' | 'file' | null;

const VideoPlayerLayout: React.FC = () => {
  const [videoSource, setVideoSource] = useState<string | null>(null);
  
  const [sourceType, setSourceType] = useState<VideoSourceType>(null);

  const [urlInput, setUrlInput] = useState<string>('');

  useEffect(() => {
    return () => {
      if (sourceType === 'file' && videoSource) {
        URL.revokeObjectURL(videoSource);
      }
    };
  }, [videoSource, sourceType]);


  const handleUrlSubmit = (e: React.FormEvent) => {
    e.preventDefault();
    const youtubeRegex = /(?:https?:\/\/)?(?:www\.)?(?:youtube\.com\/(?:[^\/\n\s]+\/\S+\/|(?:v|e(?:mbed)?)\/|\S*?[?&]v=)|youtu\.be\/)([a-zA-Z0-9_-]{11})/;
    const match = urlInput.match(youtubeRegex);

    if (match && match[1]) {
      const videoId = match[1];
      setVideoSource(`https://www.youtube.com/embed/${videoId}`);
      setSourceType('url');
    } else {
      alert('URL YouTube tidak valid. Silakan coba lagi.');
    }
  };


  const handleFileChange = (e: React.ChangeEvent<HTMLInputElement>) => {
    const file = e.target.files?.[0];
    if (file) {
      // Membuat URL sementara untuk file lokal
      const fileUrl = URL.createObjectURL(file);
      setVideoSource(fileUrl);
      setSourceType('file');
    }
  };

  return (
    <div className="min-h-screen flex items-center justify-center bg-gray-900 text-white p-4 sm:p-6 md:p-8">
      <div className="container mx-auto">
        {/* <h1 className="text-3xl font-bold mb-6 text-center">React Video Player</h1> */}
        <div className="flex flex-col md:flex-row gap-8">
          
          {/* Kolom Kiri*/}
          <div className="w-full md:flex-1 bg-black rounded-xl shadow-lg overflow-hidden">
            <div className="aspect-video w-full h-full">
              {!videoSource ? (
                <div className="flex items-center justify-center h-full text-gray-500">
                  <p className="text-xl">Video akan muncul di sini</p>
                </div>
              ) : sourceType === 'url' ? (
                <iframe
                  className="w-full h-full"
                  src={videoSource}
                  title="YouTube video player"
                  frameBorder="0"
                  allow="accelerometer; autoplay; clipboard-write; encrypted-media; gyroscope; picture-in-picture"
                  allowFullScreen
                ></iframe>
              ) : (
                <video
                  className="w-full h-full"
                  src={videoSource || ''}
                  controls
                  autoPlay
                ></video>
              )}
            </div>
          </div>

          {/* Kolom Kanan */}
          <div className="w-full md:w-1/3 lg:w-1/4">
            <div className="bg-gray-800 p-6 rounded-xl shadow-lg flex flex-col gap-6">
              
              {/* Input URL */}
              <form onSubmit={handleUrlSubmit}>
                <label htmlFor="video-url" className="block text-sm font-medium text-gray-300 mb-2">
                  Input YouTube URL
                </label>
                <input
                  type="text"
                  id="video-url"
                  value={urlInput}
                  onChange={(e) => setUrlInput(e.target.value)}
                  className="w-full bg-gray-700 border border-gray-600 rounded-md p-2 focus:ring-2 focus:ring-blue-500 focus:border-blue-500 outline-none transition"
                  placeholder="https://www.youtube.com/watch?v=..."
                />
                <button
                  type="submit"
                  className="mt-3 w-full bg-blue-600 hover:bg-blue-700 text-white font-bold py-2 px-4 rounded-md transition-colors duration-300"
                >
                  Load Video
                </button>
              </form>
              
              {/* Separator */}
              <div className="flex items-center">
                <hr className="flex-grow border-gray-600"/>
                <span className="px-3 text-gray-500 text-sm">ATAU</span>
                <hr className="flex-grow border-gray-600"/>
              </div>

              {/* Input Upload File */}
              <div>
                <label 
                  htmlFor="video-upload" 
                  className="w-full bg-green-600 hover:bg-green-700 text-white font-bold py-2 px-4 rounded-md transition-colors duration-300 cursor-pointer text-center block"
                >
                  Upload Video Dari Komputer
                </label>
                <input
                  type="file"
                  id="video-upload"
                  className="hidden"
                  accept="video/*"
                  onChange={handleFileChange}
                />
                <p className="text-xs text-gray-500 mt-2 text-center">
                  File video yang diunggah tidak dikirim ke server, hanya diputar secara lokal di browser Anda.
                </p>
              </div>

            </div>
          </div>
        </div>
      </div>
    </div>
  );
};

export default VideoPlayerLayout;