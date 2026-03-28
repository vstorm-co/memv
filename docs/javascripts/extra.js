// Smooth scroll offset for anchor links (accounts for sticky header)
document.addEventListener("DOMContentLoaded", function () {
  // Add intersection observer for code blocks — subtle reveal on scroll
  const codeBlocks = document.querySelectorAll(".md-typeset pre");
  if (codeBlocks.length && "IntersectionObserver" in window) {
    const observer = new IntersectionObserver(
      (entries) => {
        entries.forEach((entry) => {
          if (entry.isIntersecting) {
            entry.target.style.opacity = "1";
            entry.target.style.transform = "translateY(0)";
            observer.unobserve(entry.target);
          }
        });
      },
      { threshold: 0.1 }
    );

    codeBlocks.forEach((block) => {
      block.style.opacity = "0";
      block.style.transform = "translateY(8px)";
      block.style.transition = "opacity 0.4s ease, transform 0.4s ease";
      observer.observe(block);
    });
  }
});
