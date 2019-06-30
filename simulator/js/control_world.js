
var world;
var bodies = []; // instances of b2Body (from Box2D)
var maxSpeed = 30; //A cap on the speed of the objects so things don't get out of control.
var frame = 0;

var xPos = 0;
var yPos = 0;

var data = {
  physics: {},
  events: {},
  setup: {}
};

/////////////////////////////
//Set fixed/randomised properties
/////////////////////////////
var CO_damping = 10; //Controlled object damping
var damping = .05; //Uncontrolled object damping
var wall_elasticity = .98; //Sets elasticity
var wall_friction = 0.05; //Sets friction of the walls  (When does this affect things since the objects nearly always bounce rather than slide??)
var wall_mass = 0;
var wall_damping = 0;
var o_friction = .05;
var o_elasticity = 0.98;

//Declaring a bunch of needed box2d variables
var b2Vec2 = Box2D.Common.Math.b2Vec2,
  b2BodyDef = Box2D.Dynamics.b2BodyDef,
  b2Body = Box2D.Dynamics.b2Body,
  b2FixtureDef = Box2D.Dynamics.b2FixtureDef,
  b2World = Box2D.Dynamics.b2World,
  b2PolygonShape = Box2D.Collision.Shapes.b2PolygonShape;
b2CircleShape = Box2D.Collision.Shapes.b2CircleShape;
b2GravityController = Box2D.Dynamics.Controllers.b2GravityController;
b2TensorDampingController = Box2D.Dynamics.Controllers.b2TensorDampingController;
b2ContactListener = Box2D.Dynamics.b2ContactListener;


function Run() {
     
    fullX = 6;
    fullY = 4;

    stepSize = 1 / 60;

    world = new b2World(new b2Vec2(0,0));

    //////////////////
    //Create the balls
    ballWidth = Math.min(fullX / 16, fullY / 16);
    //print(ballWidth)
    print(cond.mass[0], cond.mass[1])

    for (var i=0; i<cond.sls.length; i++)
    {
        var sv = new b2Vec2(cond.svs[i].x, cond.svs[i].y);
        var name = 'o' + (i + 1);

        createBall(ballWidth, cond.sls[i].x, cond.sls[i].y, sv, b2Body.b2_dynamicBody, cond.mass[i], damping, o_friction, o_elasticity, {
                name: name,
                bodyType: "dynamic",
                W: ballWidth,
                H: ballWidth
        });

        data.physics[name] = {x:[cond.sls[i].x], y:[cond.sls[i].y], vx:[cond.svs[i].x], vy:[cond.svs[i].y],rotation:[0]};
    }

    /////////////////////////////////////////////
    //Create the walls, set them as static bodies

    borderWidth = Math.min(fullY / 20, fullX / 20);

    createBox(fullX / 2, borderWidth, fullX / 2, 0, b2Body.b2_staticBody, wall_mass, wall_damping, wall_elasticity, wall_friction, {
    name: "top_wall",
    bodyType: "static",
    W: fullX / 2,
    H: borderWidth
    });
    createBox(fullX / 2, borderWidth, fullX / 2, fullY, b2Body.b2_staticBody, wall_mass, wall_damping, wall_elasticity, wall_friction, {
    name: "bottom_wall",
    bodyType: "static",
    W: fullX / 2,
    H: borderWidth
    });
    createBox(borderWidth, fullY / 2, 0, fullY / 2, b2Body.b2_staticBody, wall_mass, wall_damping, wall_elasticity, wall_friction, {
    name: "left_wall",
    bodyType: "static",
    W: borderWidth,
    H: fullY / 2
    });
    createBox(borderWidth, fullY / 2, fullX, fullY / 2, b2Body.b2_staticBody, wall_mass, wall_damping, wall_elasticity, wall_friction, {
    name: "right_wall",
    bodyType: "static",
    W: borderWidth,
    H: fullY / 2
    });

    world.SetContactListener(listener);

    // print('Assigned starting locations and velocities');

    ///////////////////////
    //Add all the forces
    ///////////////////////

    //A matrix of all local forces (using the lower triangle)
    gravityControllers = [];

    for (var i = 0; i < cond.lf.length; i++)
    {
        gravityControllers.push([]);
    
        for (var j = i; j < cond.lf[i].length; j++)
        {
            if (cond.lf[i][j]!=0) {

                gravityControllers[i][j] = new b2GravityController()
                gravityControllers[i][j].G = cond.lf[i][j];
                gravityControllers[i][j].AddBody(bodies[i]);
                gravityControllers[i][j].AddBody(bodies[j]);
                world.AddController(gravityControllers[i][j]);
            }
        }
    }

    data.physics.co = [];
    data.physics.armforce = [];
    data.physics.mouse = {x:[], y:[]};

    /////////////////////////
    //Loop through simulation
    for (frame = 0; frame<cond.timeout; frame++)
    {
        onEF();
    }



    //End of simulation stuff...
    //Destroy bodies
    for (var i = 0; i < bodies.length; i++) {
        var body = bodies[i];
        
        world.DestroyBody(body);
    }
    bodies = []; // clear the object pointer vectors
    gravityControllers = [];

    return JSON.stringify(data);
}

////////////////////////////////////////////////
//Makes a round object in the box2d environment
////////////////////////////////////////////////
function createBall(r, x, y, starting_vec, type, density, damping, friction, restitution, userData) {
    // Create the fixture definition
    var fixDef = new b2FixtureDef;

    fixDef.density = density; // Set the density
    fixDef.friction = friction; // Set the friction
    fixDef.restitution = restitution; // Set the restitution - bounciness

    // Define the shape of the fixture
    fixDef.shape = new b2CircleShape;
    fixDef.shape.SetRadius(r);

    // Create the body definition
    var bodyDef = new b2BodyDef;
    bodyDef.type = type;

    // Set the position of the body
    bodyDef.position.x = x;
    bodyDef.position.y = y;

    // Create the body in the box2d world
    var b = world.CreateBody(bodyDef);
    test = b.CreateFixture(fixDef);

    if (typeof userData !== 'undefined') {
    b.SetUserData(userData);
    }

    //this workaround seems to do the trick
    b.m_linearDamping = damping;

    b.SetLinearVelocity(starting_vec);
    //.ApplyImpulse(up, bodies[i].GetWorldCenter());
    
    // print(test.m_density, test.m_friction, test.m_restitution)

    bodies.push(b);


    return b;
}

////////////////////////////////////////////////
//Makes a square object in the box2d environment
////////////////////////////////////////////////
function createBox(w, h, x, y, type, density, damping, friction, restitution, userData) {

  // Create the fixture definition
  var fixDef = new b2FixtureDef;

  fixDef.density = density; // Set the density
  fixDef.friction = friction; // Set the friction 
  fixDef.restitution = restitution; // Set the restitution - elasticity

  // Define the shape of the fixture
  fixDef.shape = new b2PolygonShape;
  fixDef.shape.SetAsBox(
    w // input should be half the width
    , h // input should be half the height 
  );

  // Create the body definition
  var bodyDef = new b2BodyDef;
  bodyDef.type = type;

  // Set the position of the body
  bodyDef.position.x = x;
  bodyDef.position.y = y;


  // Create the body in the box2d world
  var b = world.CreateBody(bodyDef);
  test = b.CreateFixture(fixDef);

  //What is userData exactly, and how do we use it?
  if (typeof userData !== 'undefined') {
    b.SetUserData(userData);
  }

  b.m_linearDamping = damping;
  
  // print(test.m_density, test.m_friction, test.m_restitution)

  bodies.push(b);

  return b;
}







function getSensorContact(contact) {
  var fixtureA = contact.GetFixtureA();
  var fixtureB = contact.GetFixtureB();

  var sensorA = fixtureA.IsSensor();
  var sensorB = fixtureB.IsSensor();

  if (!(sensorA || sensorB))
    return false;

  var bodyA = fixtureA.GetBody();
  var bodyB = fixtureB.GetBody();

  if (sensorA) { // bodyB should be added/removed to the buoyancy controller
    return {
      sensor: bodyA,
      body: bodyB
    };
  } else { // bodyA should be added/removed to the buoyancy controller
    return {
      sensor: bodyB,
      body: bodyA
    };
  }
}

function RandomLocations(n) {
  var array = [];
  var ball_radius = 0.25

  fullX = 6;
  fullY = 4;
  for (var i = 0; i < n; i++) {
    var okLoc = false;
    var timeout = 0;

    while (okLoc == false & timeout < 250) {

      timeout = timeout + 1;
      proposal = [Math.random() * fullX, Math.random() * fullY];
      okLoc = true;

      //Check they are not within a ball width of the edge
      if (proposal[0] < (2 * ball_radius) | proposal[0] > (fullX - (2 * ball_radius)) |
        proposal[1] < (2 * ball_radius) | proposal[1] > (fullY - (2 * ball_radius))) {
        // console.log('Too near edge', proposal);
        okLoc = false;
      }
      //Check they don't overlap
      if (i>0){
        for (var j = 0; j < array.length; j++) {

          if ((proposal[0] - array[j][0]) < ball_radius &
            (proposal[0] - array[j][0]) > (-ball_radius) &
            (proposal[1] - array[j][1]) < ball_radius &
            (proposal[1] - array[j][1]) > (-ball_radius)) {
              // console.log('conflict', array[j], proposal);
              okLoc = false;
          }
        }
      }

    }
    array.push(proposal)
  }

  return array;
}


function restartPositionsAndVelocities(){
  newLocations = RandomLocations(4)

  for (var i = 0; i < bodies.length; i++) {

    if (bodies[i].m_userData.bodyType === "dynamic")
    {
        var body = bodies[i];
        var p = body.GetPosition();
        var name = body.m_userData.name;//
        var newPosition = new b2Vec2(newLocations[i][0], newLocations[i][1])
        var newVelocity = new b2Vec2((Math.random() * 20) -10, (Math.random() * 20) -10)
        body.SetPosition(newPosition)
        body.SetLinearVelocity(newVelocity)
    }
}
}


function areBodiesTooSlow(){
    for (var i = 0; i < bodies.length; i++) {

        if (bodies[i].m_userData.bodyType === "dynamic")
        {
            var body = bodies[i];
            var p = body.GetPosition();
            var name = body.m_userData.name;//

            var vx_array = data.physics[name].vx
            var vx = vx_array[vx_array.length - 1]
            var vy_array = data.physics[name].vy
            var vy = vy_array[vy_array.length - 1]
            var speed = Math.sqrt(Math.pow(vx, 2) + Math.pow(vy, 2))
            if (speed > 0.25){
                return false
            }
        }
    }
    return true
}


function onEF() {
  //Step the world forward
  world.Step(stepSize, 3, 3);
  world.ClearForces();

  // xPos = control_path.x[frame];//e.target.mouseX;
  // yPos = control_path.y[frame];//e.target.mouseY;
  xPos = 0
  yPos = 0
  

  // data.physics.co.push(control_path.obj[frame]);
  data.physics.co.push(0)
  data.physics.mouse.x.push(xPos);
  data.physics.mouse.y.push(yPos);

  for (var i = 0; i < bodies.length; i++) {

    if (bodies[i].m_userData.bodyType === "dynamic")
    {
        var body = bodies[i];
        var p = body.GetPosition();
        var name = body.m_userData.name;//

        data.physics[name].x.push(Math.round(p.x * 100000) / 100000);
        data.physics[name].y.push(Math.round(p.y * 100000) / 100000);
        data.physics[name].vx.push(Math.round(body.m_linearVelocity.x * 100000) / 100000);
        data.physics[name].vy.push(Math.round(body.m_linearVelocity.y * 100000) / 100000);
        data.physics[name].rotation.push(Math.round(body.GetAngle() * 100000) / 100000);
        // bodies[i].SetAngle(0)
        // bodies[i].m_angularVelocity = 0
        // print(bodies[i].m_angularVelocity)
    }
}

    if (areBodiesTooSlow()){
        restartPositionsAndVelocities()
        print("Simulation with restart")
    }
  //////////////////////////////
  //Update the controlled object

  
   /* if (control_path.obj[frame]!=0) {
        var body = bodies[control_path.obj[frame]-1]; //Select which object is under control
        body.m_linearDamping = CO_damping;

        //Intervene on the physics
        var tmp = body.GetLinearVelocity();
        var xCO = body.GetPosition().x; //Position of controlled object
        var yCO = body.GetPosition().y; //Position of controlled object
        var xVec = .2*(xPos - xCO);//fistSpeed; 
        var yVec = .2*(yPos - yCO);//fistSpeed;    
        var armForce = new b2Vec2(xVec, yVec);
        
        print(frame, 'armforce!', xCO, yCO, xVec, yVec)

        data.physics.armforce.push(armForce)

        body.ApplyImpulse(armForce, body.GetWorldCenter());

        //Go back to normal damping when control released
        if (frame!=(control_path.length-1))
        {
            // print(frame)
            if (control_path.obj[frame+1]==0)
            {
                body.m_linearDamping = damping;
            }
        }

    } */


}





var listener = new b2ContactListener();
listener.BeginContact = function(contact) {
    var tmp = [];
    var tmp2 = [contact.GetFixtureA().GetBody().GetUserData(),
    contact.GetFixtureB().GetBody().GetUserData()];//.sort(); //Contact entities
    //DO STUFF WITH CONTACT HERE
};

listener.EndContact = function(contact) {
    var contactEntities = getSensorContact(contact);
    
    // print('collision! (listener)', contact);
    
    if (contactEntities) {
        var sensor = contactEntities.sensor;
        if (sensor.GetUserData()) {
            var userData = sensor.GetUserData();
            if (userData.controller) {
                userData.controller.RemoveBody(contactEntities.body);
            }
        }
    }
};
