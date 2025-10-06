// src/components/VideoPlayerLayout.tsx

import React, { useState, useEffect, useRef } from "react";
// --- BARU ---
import { io, Socket } from "socket.io-client";

// Tipe untuk sumber video agar lebih jelas
type VideoSourceType = "url" | "file" | null;

// --- BARU ---
// Definisikan alamat backend di satu tempat
const BACKEND_URL = "http://127.0.0.1:8000";

const VideoPlayerLayout: React.FC = () => {
  // State untuk menyimpan URL video yang akan diputar
  const [videoSource, setVideoSource] = useState<string | null>(null);
  // --- BARU ---
  // State untuk menyimpan URL YouTube mentah yang akan dikirim ke backend
  const [rawYoutubeUrl, setRawYoutubeUrl] = useState<string>("");
  // State untuk menyimpan tipe input (URL atau file)
  const [sourceType, setSourceType] = useState<VideoSourceType>(null);
  // State untuk mengontrol input field URL
  const [urlInput, setUrlInput] = useState<string>("");

  // State untuk status pemrosesan CCTV
  const [isProcessing, setIsProcessing] = useState<boolean>(false);
  // State untuk status download
  const [isDownloading, setIsDownloading] = useState<boolean>(false);

  // --- BARU ---
  // State untuk menyimpan frame gambar yang diproses dari backend
  const [processedFrame, setProcessedFrame] = useState<string>("");
  // State untuk menyimpan pesan status dari backend
  const [statusMessage, setStatusMessage] = useState<string>(
    "Menunggu input video..."
  );
  // Ref untuk menyimpan instance socket agar tidak dibuat ulang setiap render
  const socketRef = useRef<Socket | null>(null);

  // --- BARU ---
  // Efek untuk mengelola koneksi Socket.IO
  useEffect(() => {
    // Hubungkan ke server backend saat komponen dimuat
    socketRef.current = io(BACKEND_URL);
    const socket = socketRef.current;

    socket.on("connect", () => {
      console.log("✅ Berhasil terhubung ke server backend!");
      setStatusMessage("Terhubung. Silakan muat video.");
    });

    socket.on("disconnect", () => {
      console.log("🔌 Terputus dari server backend.");
      setStatusMessage("Koneksi terputus. Coba refresh halaman.");
      setIsProcessing(false); // Hentikan status processing jika koneksi putus
    });

    socket.on("processing_started", (data) => {
      console.log(data.message);
      setStatusMessage("Live processing sedang berjalan...");
    });

    socket.on("video_frame", (data) => {
      // Menerima frame baru dan menyimpannya di state
      // console.log("Menerima frame dari backend:", data);

      setProcessedFrame(`data:image/jpeg;base64,${data.image}`);
    });

    socket.on("error", (data) => {
      console.error("Error dari backend:", data.message);
      setStatusMessage(`Error: ${data.message}`);
      setIsProcessing(false); // Hentikan processing jika ada error
    });

    // Cleanup: putuskan koneksi saat komponen di-unmount
    return () => {
      socket.disconnect();
    };
  }, []);

  // Efek untuk membersihkan object URL dari file
  useEffect(() => {
    return () => {
      if (sourceType === "file" && videoSource) {
        URL.revokeObjectURL(videoSource);
      }
    };
  }, [videoSource, sourceType]);

  /**
   * Meng-handle submit dari input URL YouTube.
   */
  const handleUrlSubmit = (e: React.FormEvent) => {
    e.preventDefault();
    const youtubeRegex =
      /(?:https?:\/\/)?(?:www\.)?(?:youtube\.com\/(?:[^\/\n\s]+\/\S+\/|(?:v|e(?:mbed)?)\/|\S*?[?&]v=)|youtu\.be\/)([a-zA-Z0-9_-]{11})/;
    const match = urlInput.match(youtubeRegex);

    if (match && match[1]) {
      const videoId = match[1];
      setVideoSource(`https://www.youtube.com/embed/${videoId}`);

      // Simpan URL asli untuk dikirim ke backend
      setRawYoutubeUrl(urlInput);
      setSourceType("url");
      // Hentikan pemrosesan jika sedang berjalan video lain
      if (isProcessing) {
        setIsProcessing(false);
      }
      setStatusMessage("Video siap. Klik 'Run Smart CCTV' untuk memulai.");
    } else {
      alert("URL YouTube tidak valid. Silakan coba lagi.");
    }
  };

  const handleToggleProcessing = () => {
    if (sourceType !== "url") {
      alert("Fitur Smart CCTV hanya tersedia untuk video dari YouTube.");
      return;
    }

    const socket = socketRef.current;
    if (!socket) return;

    if (isProcessing) {
      socket.disconnect();
      setIsProcessing(false);
      setStatusMessage(
        "Processing dihentikan. Hubungkan kembali untuk memulai lagi."
      );
      setTimeout(() => socket.connect(), 100);
    } else {
      // Jika tidak berjalan, mulai
      if (rawYoutubeUrl) {
        socket.emit("process_youtube_url", { url: rawYoutubeUrl });
        setIsProcessing(true);
        setStatusMessage("Mengirim request ke server...");
      } else {
        alert("URL YouTube tidak ditemukan. Harap muat video terlebih dahulu.");
      }
    }
  };

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
   */
  const handleFileChange = (e: React.ChangeEvent<HTMLInputElement>) => {
    const file = e.target.files?.[0];
    if (file) {
      const fileUrl = URL.createObjectURL(file);
      setVideoSource(fileUrl);
      setSourceType("file");
      if (isProcessing) setIsProcessing(false); // Hentikan pemrosesan jika beralih ke file
    }
  };

  return (
    <div className="min-h-screen flex items-center justify-center bg-gray-900 text-white p-4 sm:p-6 md:p-8">
      <div className="container mx-auto">
        <div className="flex flex-col md:flex-row gap-8">
          {/* Kolom Kiri: Output Video */}
          <div className="w-full md:flex-1 bg-black rounded-xl shadow-lg overflow-hidden">
            <div className="aspect-video w-full h-full">
              {isProcessing ? (
                // Jika sedang diproses, tampilkan stream gambar dari backend
                <img
                  src={processedFrame}
                  alt="Live CCTV Processing..."
                  className="w-full h-full object-contain"
                />
              ) : !videoSource ? (
                // Jika tidak ada video sama sekali
                <div className="flex items-center justify-center h-full text-gray-500">
                  <p className="text-xl">Video akan muncul di sini</p>
                </div>
              ) : sourceType === "url" ? (
                // Jika ada video URL (tampilan iframe sebelum diproses)
                <iframe
                  className="w-full h-full"
                  src={videoSource}
                  title="YouTube video player"
                  frameBorder="0"
                  allow="accelerometer; autoplay; clipboard-write; encrypted-media; gyroscope; picture-in-picture"
                  allowFullScreen
                ></iframe>
              ) : (
                // Jika video dari file
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

              <div className="flex items-center">
                <hr className="flex-grow border-gray-600" />
                <span className="px-3 text-gray-500 text-sm">ATAU</span>
                <hr className="flex-grow border-gray-600" />
              </div>

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
                  File video yang diunggah tidak dikirim ke server.
                </p>
              </div>
            </div>
            <div className="bg-gray-800 p-6 rounded-xl shadow-lg flex flex-col gap-6 mt-6">
              <div className="text-center text-sm text-gray-400">
                {statusMessage}
              </div>

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
                  disabled={sourceType !== "url"}
                  className={`w-full font-bold py-2 px-4 rounded-md transition-colors duration-300 ${
                    isProcessing
                      ? "bg-red-600 hover:bg-red-700"
                      : "bg-indigo-600 hover:bg-indigo-700"
                  } disabled:bg-gray-700 disabled:text-gray-500 disabled:cursor-not-allowed`}
                >
                  {isProcessing ? "Stop Processing" : "Run Smart CCTV"}
                </button>
              </div>

              {/* Tombol Download */}
              <div className="text-center">
                <button
                  onClick={handleDownloadClip}
                  disabled={
                    sourceType !== "url" || isProcessing || isDownloading
                  }
                  className={`w-full font-bold py-2 px-4 rounded-md transition-colors duration-300 ${
                    isDownloading
                      ? "bg-gray-700 cursor-not-allowed"
                      : "bg-green-600 hover:bg-green-700"
                  } disabled:bg-gray-700 disabled:text-gray-500 disabled:cursor-not-allowed`}
                >
                  {isDownloading ? "Processing..." : "Download 30s Clip"}
                </button>
              </div>
            </div>
          </div>
        </div>
      </div>
    </div>
  );
};

export default VideoPlayerLayout;
