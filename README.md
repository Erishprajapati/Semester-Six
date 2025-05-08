# 🌄 Peak Times

**Peak Times** is a web-based application designed to assist tourists and locals in Nepal by providing location-based travel recommendations, crowd analysis, and a dynamic interactive map interface. The system helps users discover popular destinations across Nepal, understand current crowd levels, and explore similar places based on preferences and categories.

---

## 🚀 Project Overview

When tourists visit Nepal, they often struggle with identifying where to go and what to explore. *Peak Times* solves this by providing:
- A location-aware map interface
- Search functionality by district or category
- Crowd-level visualizations using bar graphs
- Content-based filtering to recommend similar places
- Informative pages with descriptions, images, and mini maps

---

## 🧭 Features

- **User Authentication**  
  🔐 Secure user registration and login using Django's built-in `User` model.

- **Location Access**  
  📍 Upon login, the user is prompted to share their location.  
  - If accepted: their current location is shown on the map  
  - If denied: they can manually input a location

- **Search by District/Category**  
  🏞️ Users can search places by district or by category (e.g., nightlife, heritage, religious, nature).

- **Crowd Data Visualization**  
  📊 Each district/place shows a bar graph representing crowd levels at popular places (dummy data for Nepal).

- **Place Recommendations**  
  🎯 Based on the selected place's category (content-based filtering), users get recommendations for similar places.

- **Place Information Page**  
  🗺️ When clicking on a bar graph:
  - Displays a mini-map to the location
  - Includes a description, image, and other relevant info

---

## 🧱 Tech Stack

| Technology      | Role                                |
|----------------|--------------------------------------|
| **Django**      | Backend framework & ORM              |
| **PostgreSQL**  | Database                             |
| **Leaflet.js**  | Interactive maps                     |
| **Chart.js**    | Bar graph for crowd data             |
| **HTML/CSS/JS** | Frontend                             |

---

## 📂 Database Schema (Simplified)

- `User`: Handles authentication (`username`, `email`, `password`)
- `Place`: Information about tourist spots (e.g., Thamel, Basantapur)
- `CrowdData`: Stores crowd levels (simulated/dummy)
- `Tag`: Categories like nightlife, historical, spiritual
- `UserLocation`: Stores user location (manual or auto-detected)

---

## 🔄 Workflow

1. **User visits** the site and registers/logs in
2. **Map access prompt** appears (Allow/Deny)
3. **User location** is set (auto or manual)
4. **User searches** by district or category
5. **Crowd levels** shown as bar graphs
6. **Clicking on a bar** opens detailed info + recommendations

---

## 📸 Screenshots

> Add UI screenshots here like:
- Login/Register Page
- Map Interface
- Crowd Bar Graph
- Recommendation View

---

## 📌 Dummy Data Note

Due to limitations in real-time data availability in Nepal, dummy data is used to simulate crowd levels and places for each district.

---

## 🧑‍💻 Developer

**Erish Prajapati**  
📧 [irishmjn@gmail.com](mailto:irishmjn@gmail.com)  
🌐 [LinkedIn](https://www.linkedin.com/in/eris-prajapati-a744ba259/)  
💻 [GitHub](https://github.com/Erishprajapati)

---

## 📜 License

This project is built for academic and non-commercial use. For other use cases, please contact the developer.

---

## ✅ Future Improvements

- Real-time crowd tracking with IoT integration or location APIs
- Mobile version of the app
- Admin dashboard for place/crowd data management
