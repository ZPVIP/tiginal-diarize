class TiginalDiarize < Formula
  desc "Fast, local speaker diarization CLI based on NVIDIA Nemotron-3"
  homepage "https://github.com/ZPVIP/tiginal-diarize"
  url "https://github.com/ZPVIP/tiginal-diarize.git", branch: "main"
  version "0.1.0"
  license "MIT"

  depends_on "rust" => :build

  def install
    system "cargo", "install", *std_cargo_args
  end

  test do
    assert_match "tiginal-diarize 0.1.0", shell_output("#{bin}/tiginal-diarize --version")
  end
end
