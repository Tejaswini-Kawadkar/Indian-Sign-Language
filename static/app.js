function playNextVideo() {
    if (index < videoPaths.length) {
        if (videoPaths[index] === 'pause') {
            // If it's a pause (space), wait for 5 seconds before playing the next video
            setTimeout(() => {
                index++;
                playNextVideo();
            }, 2000);  // 5-second gap
        } else {
            // Play the video
            videoPlayer.src = '/video/' + videoPaths[index].split('/').pop();  // Extract filename
            videoPlayer.play();
            videoPlayer.onended = function () {
                index++;
                playNextVideo();
            };
        }
    } else {
        // Reset the index after all videos have been played (optional)
        index = 0; // Reset for future plays, if needed
    }
}

playNextVideo();  // Start playing the first video

function playVideo(videoPaths) {
    let videoPlayer = document.getElementById('video-player');
    let index = 0;

    function playNextVideo() {
        if (index < videoPaths.length) {
            if (videoPaths[index] === 'pause') {
                // If it's a pause (space), wait for 5 seconds before playing the next video
                setTimeout(() => {
                    index++;
                    playNextVideo();
                }, 1000);  // 2-second gap
            } else {
                // Play the video
                videoPlayer.src = '/video/' + videoPaths[index].split('/').pop();  // Extract filename
                videoPlayer.play();
                videoPlayer.onended = function () {
                    index++;
                    playNextVideo();
                };
            }
        } else {
            // Reset the index after all videos have been played (optional)
            index = 0; // Reset for future plays, if needed
        }
    }

    playNextVideo();  // Start playing the first video
}

document.getElementById('text-form').addEventListener('submit', function (e) {
    e.preventDefault();
    let inputText = document.getElementById('text-input').value;
    fetch('/text_input', {
        method: 'POST',
        headers: {
            'Content-Type': 'application/x-www-form-urlencoded',
        },
        body: 'input_text=' + encodeURIComponent(inputText),
    })
    .then(response => response.json())
    .then(data => {
        if (data.videos) {
            playVideo(data.videos);  // Play all videos in sequence
        } else if (data.error) {
            alert(data.error);
        }
    });
});


