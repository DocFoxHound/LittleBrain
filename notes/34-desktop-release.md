# 34. LittleBrain desktop release

The Linux desktop release packages the signal and danger games from the public repository. Run `littlebrain` for the signal game or `littlebrain --game danger` for the danger game. Space pauses the viewer. Hover over a neuron to inspect it.

Build from the repository root with `bash packaging/build-linux.sh`. The script uses a temporary Python environment, bundles the viewer with its CPU dependencies, and makes the executable play two turns of the signal game headlessly. The build succeeds only if the viewer writes the expected screenshot in the designated save folder. The packaged viewer grows a fresh brain each launch. Screenshots taken under SLAPorSLOP go in the player's save folder.

A version tag triggers Slop Build for Linux. Its archive launches `littlebrain`, and its Debian package is available for system installation. The workflow requests GitHub build attestations and submits the release to the LittleBrain store draft. The publisher reviews the result on SLAPorSLOP. [Publishing guide](https://slaporslop.games/publishing.md)
