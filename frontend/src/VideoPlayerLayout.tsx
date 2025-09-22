// src/components/VideoPlayerLayout.tsx

import React, { useState, useEffect } from "react";

// Tipe untuk sumber video agar lebih jelas
type VideoSourceType = "url" | "file" | null;

const VideoPlayerLayout: React.FC = () => {
  // State untuk menyimpan URL video yang akan diputar
  const [videoSource, setVideoSource] = useState<string | null>(null);

  // State untuk menyimpan tipe input (URL atau file)
  const [sourceType, setSourceType] = useState<VideoSourceType>(null);

  // State untuk mengontrol input field URL
  const [urlInput, setUrlInput] = useState<string>("");

  // State baru untuk status pemrosesan CCTV
  const [isProcessing, setIsProcessing] = useState<boolean>(false);
  // State baru untuk status download
  const [isDownloading, setIsDownloading] = useState<boolean>(false);

  // Fungsi untuk membersihkan object URL dari file yang diunggah untuk mencegah memory leak
  useEffect(() => {
    // Cleanup function ini akan dijalankan saat komponen unmount atau saat videoSource berubah
    return () => {
      if (sourceType === "file" && videoSource) {
        URL.revokeObjectURL(videoSource);
      }
    };
  }, [videoSource, sourceType]);

  /**
   * Meng-handle submit dari input URL YouTube.
   * Fungsi ini akan mengubah URL YouTube standar menjadi URL embed.
   */
  const handleUrlSubmit = (e: React.FormEvent) => {
    e.preventDefault();
    // Regex untuk mengekstrak ID video dari berbagai format URL YouTube
    const youtubeRegex =
      /(?:https?:\/\/)?(?:www\.)?(?:youtube\.com\/(?:[^\/\n\s]+\/\S+\/|(?:v|e(?:mbed)?)\/|\S*?[?&]v=)|youtu\.be\/)([a-zA-Z0-9_-]{11})/;
    const match = urlInput.match(youtubeRegex);

    if (match && match[1]) {
      const videoId = match[1];
      setVideoSource(`https://www.youtube.com/embed/${videoId}`);
      setSourceType("url");
    } else {
      alert("URL YouTube tidak valid. Silakan coba lagi.");
    }
  };
  // Handler untuk tombol Run/Stop Smart CCTV
  const handleToggleProcessing = () => {
    if (!videoSource) {
      alert("Silakan muat video terlebih dahulu.");
      return;
    }
    setIsProcessing((prev) => !prev);
  };

  // Handler untuk tombol download (simulasi)
  const handleDownloadClip = () => {
    if (sourceType !== "url") {
      alert("Fungsi download hanya tersedia untuk video dari YouTube.");
      return;
    }
    setIsDownloading(true);
    console.log(
      "SIMULASI: Mengirim request ke backend untuk download & potong video dari:",
      videoSource
    );

    // Simulasi proses backend yang memakan waktu 4 detik
    setTimeout(() => {
      setIsDownloading(false);
      alert(
        "Simulasi Selesai: Klip 30 detik siap diunduh! (Ini memerlukan implementasi backend)."
      );
    }, 4000);
  };

  /**
   * Meng-handle perubahan pada input file.
   * Fungsi ini akan membuat Object URL dari file yang dipilih.
   */
  const handleFileChange = (e: React.ChangeEvent<HTMLInputElement>) => {
    const file = e.target.files?.[0];
    if (file) {
      // Membuat URL sementara untuk file lokal
      const fileUrl = URL.createObjectURL(file);
      setVideoSource(fileUrl);
      setSourceType("file");
    }
  };

  return (
    <div className="min-h-screen flex items-center justify-center bg-gray-900 text-white p-4 sm:p-6 md:p-8">
      <div className="container mx-auto">
        {/* <h1 className="text-3xl font-bold mb-6 text-center">React Video Player</h1> */}
        <div className="flex flex-col md:flex-row gap-8">
          {/* Kolom Kiri: Output Video */}
          <div className="w-full md:flex-1 bg-black rounded-xl shadow-lg overflow-hidden">
            <div className="aspect-video w-full h-full">
              {!videoSource ? (
                <div className="flex items-center justify-center h-full text-gray-500">
                  <p className="text-xl">Video akan muncul di sini</p>
                </div>
              ) : sourceType === "url" ? (
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
                  src={videoSource || ""}
                  controls
                  autoPlay
                ></video>
              )}
            </div>
          </div>

          {/* Kolom Kanan: Input Kontrol */}
          <div className="w-full md:w-1/3 lg:w-1/4">
            <div className="bg-gray-800 p-6 rounded-xl shadow-lg flex flex-col gap-6">
              {/* Input via URL */}
              <form onSubmit={handleUrlSubmit}>
                <label
                  htmlFor="video-url"
                  className="block text-sm font-medium text-gray-300 mb-2"
                >
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
                <hr className="flex-grow border-gray-600" />
                <span className="px-3 text-gray-500 text-sm">ATAU</span>
                <hr className="flex-grow border-gray-600" />
              </div>

              {/* Input via Upload File */}
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
                  File video yang diunggah tidak dikirim ke server, hanya
                  diputar secara lokal di browser Anda.
                </p>
              </div>
            </div>
            <div className="bg-gray-800 p-6 rounded-xl shadow-lg flex flex-col gap-6 mt-6">
              {/* Tombol Running */}
              <div className="text-center">
                {isProcessing && (
                  <div className="flex items-center justify-center gap-2 mb-2">
                    <span className="relative flex h-3 w-3">
                      <span className="animate-ping absolute inline-flex h-full w-full rounded-full bg-red-400 opacity-75"></span>
                      <span className="relative inline-flex rounded-full h-3 w-3 bg-red-500"></span>
                    </span>
                    <span className="text-red-400 font-semibold">
                      LIVE PROCESSING
                    </span>
                  </div>
                )}
                <button
                  onClick={handleToggleProcessing}
                  className={`w-full font-bold py-2 px-4 rounded-md transition-colors duration-300 ${
                    isProcessing
                      ? "bg-red-600 hover:bg-red-700"
                      : "bg-indigo-600 hover:bg-indigo-700"
                  }`}
                >
                  {isProcessing ? "Stop Processing" : "Run Smart CCTV"}
                </button>
              </div>

              {/* Tombol Download */}
              <div>
                <button
                  onClick={handleDownloadClip}
                  disabled={sourceType !== "url" || isDownloading}
                  className="w-full bg-gray-600 hover:bg-gray-500 text-white font-bold py-2 px-4 rounded-md transition-colors duration-300 disabled:bg-gray-800 disabled:text-gray-500 disabled:cursor-not-allowed"
                >
                  {isDownloading ? "Processing..." : "Download 30s Clip"}
                </button>
                <p className="text-xs text-gray-400 mt-2 text-center">
                  Hanya berfungsi untuk video YouTube.
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
