const form = document.querySelector('#predictionForm');
const imageInput = document.querySelector('#imageInput');
const dropZone = document.querySelector('#dropZone');
const previewWrap = document.querySelector('#previewWrap');
const previewImage = document.querySelector('#previewImage');
const fileName = document.querySelector('#fileName');
const detectButton = document.querySelector('#detectButton');
const resetButton = document.querySelector('#resetButton');
const statusMessage = document.querySelector('#statusMessage');
const resultsSection = document.querySelector('#resultsSection');
const originalImage = document.querySelector('#originalImage');
const detectedImage = document.querySelector('#detectedImage');
const packetCount = document.querySelector('#packetCount');
const heroCarousel = document.querySelector('#heroCarousel');
const carouselSlides = [...document.querySelectorAll('.carousel-slide')];
const carouselDots = [...document.querySelectorAll('.carousel-dot')];

let selectedFile = null;
let previewUrl = null;
let carouselIndex = 0;
let carouselTimer = null;

function showCarouselSlide(index) {
  carouselIndex = index % carouselSlides.length;
  carouselSlides.forEach((slide, slideIndex) => slide.classList.toggle('is-active', slideIndex === carouselIndex));
  carouselDots.forEach((dot, dotIndex) => dot.classList.toggle('is-active', dotIndex === carouselIndex));
}

function startCarousel() {
  carouselTimer = window.setInterval(() => showCarouselSlide(carouselIndex + 1), 4800);
}

if (heroCarousel && carouselSlides.length) {
  startCarousel();
  heroCarousel.addEventListener('mouseenter', () => window.clearInterval(carouselTimer));
  heroCarousel.addEventListener('mouseleave', startCarousel);
  heroCarousel.addEventListener('focusin', () => window.clearInterval(carouselTimer));
  heroCarousel.addEventListener('focusout', startCarousel);
}

function setStatus(message, type = '') {
  statusMessage.textContent = message;
  statusMessage.className = `status ${type}`;
}

function isValidImage(file) {
  return file && ['image/jpeg', 'image/png'].includes(file.type) && file.size <= 10 * 1024 * 1024;
}

function selectFile(file) {
  if (!isValidImage(file)) {
    setStatus('Please choose a JPG, JPEG, or PNG image smaller than 10 MB.', 'error');
    return;
  }

  if (previewUrl) URL.revokeObjectURL(previewUrl);
  selectedFile = file;
  previewUrl = URL.createObjectURL(file);
  previewImage.src = previewUrl;
  fileName.textContent = file.name;
  previewWrap.hidden = false;
  detectButton.disabled = false;
  resetButton.disabled = false;
  resultsSection.hidden = true;
  setStatus('Image ready for detection.', 'ready');
}

function resetForm() {
  if (previewUrl) URL.revokeObjectURL(previewUrl);
  selectedFile = null;
  previewUrl = null;
  imageInput.value = '';
  previewWrap.hidden = true;
  resultsSection.hidden = true;
  detectButton.disabled = true;
  resetButton.disabled = true;
  setStatus('');
}

imageInput.addEventListener('change', (event) => selectFile(event.target.files[0]));
resetButton.addEventListener('click', resetForm);

['dragenter', 'dragover'].forEach((eventName) => dropZone.addEventListener(eventName, (event) => {
  event.preventDefault();
  dropZone.classList.add('is-dragging');
}));

dropZone.addEventListener('dragleave', () => dropZone.classList.remove('is-dragging'));
dropZone.addEventListener('drop', (event) => {
  event.preventDefault();
  dropZone.classList.remove('is-dragging');
  selectFile(event.dataTransfer.files[0]);
});

form.addEventListener('submit', async (event) => {
  event.preventDefault();
  if (!selectedFile) return;

  const data = new FormData();
  data.append('image', selectedFile);
  detectButton.disabled = true;
  resetButton.disabled = true;
  detectButton.querySelector('span').textContent = 'Detecting...';
  setStatus('Running YOLOv11 detection locally...', 'loading');

  try {
    const response = await fetch('/predict', { method: 'POST', body: data });
    const result = await response.json();
    if (result.code === 'not_gold_packet') {
      resultsSection.hidden = true;
      setStatus(result.message, 'error');
      return;
    }
    if (!response.ok || !result.success) throw new Error(result.message || 'Detection failed.');

    originalImage.src = result.original_url;
    detectedImage.src = result.detected_url;
    packetCount.textContent = result.count;
    resultsSection.hidden = false;
    resultsSection.scrollIntoView({ behavior: 'smooth', block: 'start' });
    setStatus('Detection finished successfully.', 'success');
  } catch (error) {
    setStatus(error.message, 'error');
  } finally {
    detectButton.disabled = false;
    resetButton.disabled = false;
    detectButton.querySelector('span').textContent = 'Detect packets';
  }
});
