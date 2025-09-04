document.addEventListener('DOMContentLoaded', function() {
    const videoFeed = document.getElementById('video-feed');

    function updateVideoSource() {
        const videoSource = videoFeed.src;
        videoFeed.src = videoSource.split('?')[0] + '?' + new Date().getTime();
    }

    setInterval(updateVideoSource, 100); // Update every 100ms
});